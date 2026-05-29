//! BioCLIP taxon catalog: a memory-mapped matrix of precomputed text
//! embeddings + taxonomy labels, with an optional HNSW index for ANN search.
//!
//! `top_k` does ANN search (when an index is present) to gather candidates, then
//! reranks them by exact cosine similarity and returns softmax-scored labels.
//! This is framework-agnostic — the only Burn-side input is the image embedding.

use std::{
    fs::File,
    path::{Path, PathBuf},
};

use half::f16;
use hnswlib_rs::HnswIndex;
use lumen_schema::Label;

use crate::service::{ServiceError, ServiceResult};

/// Number of ANN candidates fetched before exact rerank.
const BIOCLIP_HNSW_RERANK_K: usize = 200;
const BIOCLIP_TAXONOMY_PLACEHOLDER: &str = "*";

pub struct BioClipDataset {
    name: String,
    embeddings: MmapNpyMatrix,
    labels: Vec<String>,
    layout: BioClipEmbeddingLayout,
    index: Option<HnswIndex<'static>>,
    #[allow(dead_code)]
    mmap: Option<memmap2::Mmap>,
}

impl BioClipDataset {
    pub fn open(
        name: String,
        embeddings_path: PathBuf,
        labels_path: PathBuf,
        index_path: Option<PathBuf>,
    ) -> ServiceResult<Self> {
        let embeddings = MmapNpyMatrix::open(&embeddings_path)?;
        let labels = load_bioclip_labels(&labels_path)?;
        let layout = BioClipEmbeddingLayout::from_shape(labels.len(), &embeddings).ok_or_else(|| {
            ServiceError::InvalidArgument(format!(
                "BioCLIP dataset `{name}` label count {} from {} does not match embedding matrix shape [{}, {}] from {}",
                labels.len(),
                labels_path.display(),
                embeddings.rows(),
                embeddings.cols(),
                embeddings_path.display()
            ))
        })?;
        if layout.embedding_dim(&embeddings) == 0 {
            return Err(ServiceError::InvalidArgument(format!(
                "BioCLIP dataset `{name}` embedding dimension must be greater than zero"
            )));
        }

        let mut mmap_owner = None;
        let index = if let Some(path) = index_path {
            let file = std::fs::File::open(&path).map_err(|err| {
                ServiceError::InvalidArgument(format!(
                    "failed to open BioCLIP HNSW index at {}: {err}",
                    path.display()
                ))
            })?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|err| {
                ServiceError::InvalidArgument(format!(
                    "failed to mmap BioCLIP HNSW index at {}: {err}",
                    path.display()
                ))
            })?;
            let dim = layout.embedding_dim(&embeddings);

            // Extend the mmap reference to 'static: it is stored alongside the
            // index inside BioClipDataset and outlives it.
            let mmap_static_ref: &'static [u8] = unsafe { std::mem::transmute(&mmap[..]) };
            let mut graph = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                HnswIndex::load(dim, mmap_static_ref)
            }))
            .map_err(|_| {
                ServiceError::Internal(format!("failed to load HNSW index at {}", path.display()))
            })?;

            if graph.dim != dim {
                return Err(ServiceError::InvalidArgument(format!(
                    "BioCLIP HNSW index at {} has dimension {}, expected {}",
                    path.display(),
                    graph.dim,
                    dim
                )));
            }
            graph.ef = 256;
            mmap_owner = Some(mmap);
            Some(graph)
        } else {
            None
        };

        Ok(Self {
            name,
            embeddings,
            labels,
            layout,
            index,
            mmap: mmap_owner,
        })
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn top_k(
        &self,
        query: &[f32],
        top_k: usize,
        logit_scale: f32,
    ) -> ServiceResult<Vec<Label>> {
        let embedding_dim = self.layout.embedding_dim(&self.embeddings);
        if query.len() != embedding_dim {
            return Err(ServiceError::Internal(format!(
                "BioCLIP image embedding dimension {} does not match dataset `{}` dimension {}",
                query.len(),
                self.name,
                embedding_dim
            )));
        }
        let query_norm = l2_norm(query);
        if query_norm == 0.0 {
            return Err(ServiceError::Internal(
                "BioCLIP image embedding has zero norm".to_owned(),
            ));
        }

        let limit = top_k.min(self.labels.len());
        let best = if let Some(graph) = &self.index {
            let search_k = BIOCLIP_HNSW_RERANK_K.min(self.labels.len());
            let search_hits = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                graph.search_knn(query, search_k, hnsw_inner_product_distance)
            }))
            .map_err(|_| ServiceError::Internal("HNSW search failed".to_owned()))?;
            let candidates: Vec<usize> = search_hits.into_iter().map(|(label, _)| label).collect();
            self.rerank_candidates(query, query_norm, limit, candidates, logit_scale)?
        } else {
            match self.layout {
                BioClipEmbeddingLayout::LabelRows => {
                    self.top_k_label_rows(query, query_norm, limit, logit_scale)?
                }
                BioClipEmbeddingLayout::LabelColumns => {
                    self.top_k_label_columns(query, query_norm, limit, logit_scale)?
                }
            }
        };

        Ok(best
            .into_iter()
            .map(|(index, score)| Label {
                label: self.labels[index].clone(),
                score,
            })
            .collect())
    }

    fn top_k_label_rows(
        &self,
        query: &[f32],
        query_norm: f32,
        limit: usize,
        logit_scale: f32,
    ) -> ServiceResult<Vec<(usize, f32)>> {
        let rows = self.embeddings.rows();
        let scale = logit_scale.exp();

        let mut similarities = Vec::with_capacity(rows);
        let mut max_similarity = f32::NEG_INFINITY;
        for label_index in 0..rows {
            let sim = self
                .embeddings
                .row_cosine_similarity(label_index, query, query_norm)?;
            similarities.push(sim);
            if sim > max_similarity && sim.is_finite() {
                max_similarity = sim;
            }
        }
        softmax_top_k(&similarities, max_similarity, scale, limit)
    }

    fn rerank_candidates(
        &self,
        query: &[f32],
        query_norm: f32,
        limit: usize,
        candidates: Vec<usize>,
        logit_scale: f32,
    ) -> ServiceResult<Vec<(usize, f32)>> {
        let scale = logit_scale.exp();
        let mut candidate_sims = Vec::with_capacity(candidates.len());
        let mut max_similarity = f32::NEG_INFINITY;

        for label_index in candidates {
            if label_index >= self.labels.len() {
                continue;
            }
            let sim = match self.layout {
                BioClipEmbeddingLayout::LabelRows => {
                    self.embeddings
                        .row_cosine_similarity(label_index, query, query_norm)?
                }
                BioClipEmbeddingLayout::LabelColumns => {
                    self.embeddings
                        .column_cosine_similarity(label_index, query, query_norm)?
                }
            };
            candidate_sims.push((label_index, sim));
            if sim > max_similarity && sim.is_finite() {
                max_similarity = sim;
            }
        }

        if max_similarity == f32::NEG_INFINITY {
            return Ok(Vec::new());
        }
        let max_scaled = max_similarity * scale;
        let mut sum_exp = 0.0f32;
        for &(_, sim) in &candidate_sims {
            if sim.is_finite() {
                sum_exp += ((sim * scale) - max_scaled).exp();
            }
        }

        let mut best: Vec<(usize, f32)> = Vec::with_capacity(limit);
        for &(label_index, sim) in &candidate_sims {
            if sim.is_finite() && sum_exp > 0.0 {
                let softmax_score = ((sim * scale) - max_scaled).exp() / sum_exp;
                push_top_k(&mut best, limit, label_index, softmax_score);
            }
        }
        Ok(best)
    }

    fn top_k_label_columns(
        &self,
        query: &[f32],
        query_norm: f32,
        limit: usize,
        logit_scale: f32,
    ) -> ServiceResult<Vec<(usize, f32)>> {
        let label_count = self.labels.len();
        let mut dots = vec![0.0f32; label_count];
        let mut norms_sq = vec![0.0f32; label_count];

        for (dim, query_value) in query.iter().copied().enumerate() {
            match self.embeddings.element_type {
                NpyElementType::Float32 => {
                    for (label_index, chunk) in
                        self.embeddings.row_bytes(dim).chunks_exact(4).enumerate()
                    {
                        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        dots[label_index] += value * query_value;
                        norms_sq[label_index] += value * value;
                    }
                }
                NpyElementType::Float16 => {
                    for (label_index, chunk) in
                        self.embeddings.row_bytes(dim).chunks_exact(2).enumerate()
                    {
                        let value = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                        dots[label_index] += value * query_value;
                        norms_sq[label_index] += value * value;
                    }
                }
            }
        }

        let mut similarities = Vec::with_capacity(label_count);
        let mut max_similarity = f32::NEG_INFINITY;
        for label_index in 0..label_count {
            let norm_sq = norms_sq[label_index];
            if norm_sq == 0.0 {
                similarities.push(f32::NAN);
                continue;
            }
            let sim = dots[label_index] / (norm_sq.sqrt() * query_norm);
            similarities.push(sim);
            if sim > max_similarity && sim.is_finite() {
                max_similarity = sim;
            }
        }
        softmax_top_k(&similarities, max_similarity, logit_scale.exp(), limit)
    }
}

fn softmax_top_k(
    similarities: &[f32],
    max_similarity: f32,
    scale: f32,
    limit: usize,
) -> ServiceResult<Vec<(usize, f32)>> {
    if max_similarity == f32::NEG_INFINITY {
        return Ok(Vec::new());
    }
    let max_scaled = max_similarity * scale;
    let mut sum_exp = 0.0f32;
    for &sim in similarities {
        if sim.is_finite() {
            sum_exp += ((sim * scale) - max_scaled).exp();
        }
    }
    let mut best: Vec<(usize, f32)> = Vec::with_capacity(limit);
    for (label_index, &sim) in similarities.iter().enumerate() {
        if sim.is_finite() && sum_exp > 0.0 {
            let softmax_score = ((sim * scale) - max_scaled).exp() / sum_exp;
            push_top_k(&mut best, limit, label_index, softmax_score);
        }
    }
    Ok(best)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BioClipEmbeddingLayout {
    /// Matrix shape is `[label_count, embedding_dim]`.
    LabelRows,
    /// Matrix shape is `[embedding_dim, label_count]`.
    LabelColumns,
}

impl BioClipEmbeddingLayout {
    fn from_shape(label_count: usize, embeddings: &MmapNpyMatrix) -> Option<Self> {
        if embeddings.rows() == label_count {
            Some(Self::LabelRows)
        } else if embeddings.cols() == label_count {
            Some(Self::LabelColumns)
        } else {
            None
        }
    }

    fn embedding_dim(self, embeddings: &MmapNpyMatrix) -> usize {
        match self {
            Self::LabelRows => embeddings.cols(),
            Self::LabelColumns => embeddings.rows(),
        }
    }
}

struct MmapNpyMatrix {
    mmap: memmap2::Mmap,
    data_offset: usize,
    rows: usize,
    cols: usize,
    element_type: NpyElementType,
}

impl MmapNpyMatrix {
    fn open(path: &Path) -> ServiceResult<Self> {
        let file = File::open(path).map_err(|err| {
            ServiceError::InvalidArgument(format!(
                "failed to open BioCLIP embeddings at {}: {err}",
                path.display()
            ))
        })?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(|err| {
            ServiceError::InvalidArgument(format!(
                "failed to mmap BioCLIP embeddings at {}: {err}",
                path.display()
            ))
        })?;
        let header = parse_npy_header(&mmap, path)?;
        let expected_bytes = header
            .rows
            .checked_mul(header.cols)
            .and_then(|count| count.checked_mul(header.element_type.size_bytes()))
            .and_then(|bytes| bytes.checked_add(header.data_offset))
            .ok_or_else(|| {
                ServiceError::InvalidArgument(format!(
                    "BioCLIP embeddings at {} are too large to address",
                    path.display()
                ))
            })?;
        if mmap.len() < expected_bytes {
            return Err(ServiceError::InvalidArgument(format!(
                "BioCLIP embeddings at {} are truncated: expected at least {expected_bytes} bytes, got {}",
                path.display(),
                mmap.len()
            )));
        }

        Ok(Self {
            mmap,
            data_offset: header.data_offset,
            rows: header.rows,
            cols: header.cols,
            element_type: header.element_type,
        })
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn value(&self, row: usize, col: usize) -> f32 {
        let offset = self.data_offset + (row * self.cols + col) * self.element_type.size_bytes();
        match self.element_type {
            NpyElementType::Float32 => f32::from_le_bytes([
                self.mmap[offset],
                self.mmap[offset + 1],
                self.mmap[offset + 2],
                self.mmap[offset + 3],
            ]),
            NpyElementType::Float16 => {
                f16::from_le_bytes([self.mmap[offset], self.mmap[offset + 1]]).to_f32()
            }
        }
    }

    fn row_bytes(&self, row: usize) -> &[u8] {
        let row_size = self.cols * self.element_type.size_bytes();
        let offset = self.data_offset + row * row_size;
        &self.mmap[offset..offset + row_size]
    }

    fn row_cosine_similarity(
        &self,
        row: usize,
        query: &[f32],
        query_norm: f32,
    ) -> ServiceResult<f32> {
        let mut dot = 0.0;
        let mut row_norm_sq = 0.0;
        for (col, query_value) in query.iter().enumerate() {
            let value = self.value(row, col);
            dot += value * query_value;
            row_norm_sq += value * value;
        }
        if row_norm_sq == 0.0 {
            return Ok(f32::NAN);
        }
        Ok(dot / (row_norm_sq.sqrt() * query_norm))
    }

    fn column_cosine_similarity(
        &self,
        col: usize,
        query: &[f32],
        query_norm: f32,
    ) -> ServiceResult<f32> {
        let mut dot = 0.0;
        let mut col_norm_sq = 0.0;
        for (row, query_value) in query.iter().enumerate() {
            let value = self.value(row, col);
            dot += value * query_value;
            col_norm_sq += value * value;
        }
        if col_norm_sq == 0.0 {
            return Ok(f32::NAN);
        }
        Ok(dot / (col_norm_sq.sqrt() * query_norm))
    }
}

struct NpyHeader {
    data_offset: usize,
    rows: usize,
    cols: usize,
    element_type: NpyElementType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NpyElementType {
    Float16,
    Float32,
}

impl NpyElementType {
    fn size_bytes(self) -> usize {
        match self {
            Self::Float16 => std::mem::size_of::<f16>(),
            Self::Float32 => std::mem::size_of::<f32>(),
        }
    }
}

fn parse_npy_header(bytes: &[u8], path: &Path) -> ServiceResult<NpyHeader> {
    const MAGIC: &[u8] = b"\x93NUMPY";
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err(ServiceError::InvalidArgument(format!(
            "BioCLIP embeddings at {} are not a .npy file",
            path.display()
        )));
    }

    let major = bytes[6];
    let header_len_size = match major {
        1 => 2,
        2 | 3 => 4,
        other => {
            return Err(ServiceError::InvalidArgument(format!(
                "unsupported .npy major version {other} in {}",
                path.display()
            )));
        }
    };
    let header_len_offset = 8;
    let header_offset = header_len_offset + header_len_size;
    if bytes.len() < header_offset {
        return Err(ServiceError::InvalidArgument(format!(
            "BioCLIP embeddings at {} have an incomplete .npy header",
            path.display()
        )));
    }
    let header_len = match header_len_size {
        2 => u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
        4 => u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
        _ => unreachable!(),
    };
    let data_offset = header_offset + header_len;
    if bytes.len() < data_offset {
        return Err(ServiceError::InvalidArgument(format!(
            "BioCLIP embeddings at {} have a truncated .npy header",
            path.display()
        )));
    }
    let header = std::str::from_utf8(&bytes[header_offset..data_offset]).map_err(|err| {
        ServiceError::InvalidArgument(format!(
            "BioCLIP embeddings at {} have a non-UTF8 .npy header: {err}",
            path.display()
        ))
    })?;

    let element_type = if header.contains("'descr': '<f4'")
        || header.contains("\"descr\": \"<f4\"")
        || header.contains("'descr': '|f4'")
        || header.contains("\"descr\": \"|f4\"")
    {
        NpyElementType::Float32
    } else if header.contains("'descr': '<f2'")
        || header.contains("\"descr\": \"<f2\"")
        || header.contains("'descr': '|f2'")
        || header.contains("\"descr\": \"|f2\"")
    {
        NpyElementType::Float16
    } else {
        return Err(ServiceError::InvalidArgument(format!(
            "BioCLIP embeddings at {} must be little-endian float16 or float32 .npy",
            path.display()
        )));
    };
    if !(header.contains("'fortran_order': False")
        || header.contains("\"fortran_order\": False")
        || header.contains("\"fortran_order\": false"))
    {
        return Err(ServiceError::InvalidArgument(format!(
            "BioCLIP embeddings at {} must be C-contiguous, not Fortran order",
            path.display()
        )));
    }

    let shape_start = header.find('(').ok_or_else(|| {
        ServiceError::InvalidArgument(format!("missing .npy shape in {}", path.display()))
    })?;
    let shape_end = header[shape_start..]
        .find(')')
        .map(|index| shape_start + index)
        .ok_or_else(|| {
            ServiceError::InvalidArgument(format!("missing .npy shape in {}", path.display()))
        })?;
    let shape: Vec<usize> = header[shape_start + 1..shape_end]
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value.parse::<usize>().map_err(|err| {
                ServiceError::InvalidArgument(format!(
                    "invalid .npy shape dimension `{value}` in {}: {err}",
                    path.display()
                ))
            })
        })
        .collect::<ServiceResult<_>>()?;
    if shape.len() != 2 {
        return Err(ServiceError::InvalidArgument(format!(
            "BioCLIP embeddings at {} must be a 2D matrix, got shape {:?}",
            path.display(),
            shape
        )));
    }

    Ok(NpyHeader {
        data_offset,
        rows: shape[0],
        cols: shape[1],
        element_type,
    })
}

#[derive(serde::Deserialize)]
struct BioClipLabel {
    #[serde(default)]
    kingdom: String,
    #[serde(default)]
    phylum: String,
    #[serde(default)]
    class: String,
    #[serde(default)]
    order: String,
    #[serde(default)]
    family: String,
    #[serde(default)]
    genus: String,
    #[serde(default)]
    species: String,
    #[serde(default)]
    common_name: String,
}

fn clean_rank(s: &str) -> String {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        BIOCLIP_TAXONOMY_PLACEHOLDER.to_owned()
    } else {
        trimmed.to_owned()
    }
}

/// Loads the taxonomy catalog and renders each entry as a 8-rank path:
/// `kingdom/phylum/class/order/family/genus/species/common_name` (empty ranks
/// become `*`).
fn load_bioclip_labels(path: &Path) -> ServiceResult<Vec<String>> {
    let contents = std::fs::read_to_string(path).map_err(|err| {
        ServiceError::InvalidArgument(format!(
            "failed to read BioCLIP labels at {}: {err}",
            path.display()
        ))
    })?;
    let items: Vec<BioClipLabel> = serde_json::from_str(&contents).map_err(|err| {
        ServiceError::InvalidArgument(format!(
            "failed to parse BioCLIP labels at {}: {err}",
            path.display()
        ))
    })?;

    let labels = items
        .into_iter()
        .map(|item| {
            format!(
                "{}/{}/{}/{}/{}/{}/{}/{}",
                clean_rank(&item.kingdom),
                clean_rank(&item.phylum),
                clean_rank(&item.class),
                clean_rank(&item.order),
                clean_rank(&item.family),
                clean_rank(&item.genus),
                clean_rank(&item.species),
                clean_rank(&item.common_name),
            )
        })
        .collect();

    Ok(labels)
}

fn push_top_k(best: &mut Vec<(usize, f32)>, limit: usize, label_index: usize, score: f32) {
    if limit == 0 || !score.is_finite() {
        return;
    }
    best.push((label_index, score));
    best.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    best.truncate(limit);
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn hnsw_inner_product_distance(a: &[f32], b: &[f32]) -> f64 {
    let dot = a
        .iter()
        .zip(b.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum::<f32>();
    f64::from(1.0 - dot)
}

#[cfg(test)]
mod tests {
    use super::push_top_k;

    #[test]
    fn push_top_k_keeps_highest_scores() {
        let mut best = Vec::new();
        for (i, s) in [(0, 0.1), (1, 0.9), (2, 0.5), (3, 0.7)] {
            push_top_k(&mut best, 2, i, s);
        }
        assert_eq!(best, vec![(1, 0.9), (3, 0.7)]);
    }
}

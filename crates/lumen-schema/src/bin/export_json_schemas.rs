use std::{fs, path::Path};

use schemars::{JsonSchema, schema_for};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = Path::new("schemas");
    let result_dir = out_dir.join("result");
    let config_dir = out_dir.join("config");
    let model_dir = out_dir.join("model");

    fs::create_dir_all(&result_dir)?;
    fs::create_dir_all(&config_dir)?;
    fs::create_dir_all(&model_dir)?;

    export::<lumen_schema::EmbeddingV1>(&result_dir.join("embedding_v1.schema.json"))?;
    export::<lumen_schema::FaceV1>(&result_dir.join("face_v1.schema.json"))?;
    export::<lumen_schema::LabelsV1>(&result_dir.join("labels_v1.schema.json"))?;
    export::<lumen_schema::OCRV1>(&result_dir.join("ocr_v1.schema.json"))?;
    export::<lumen_schema::TextGenerationV1>(&result_dir.join("text_generation_v1.schema.json"))?;

    export::<lumen_schema::LumenConfig>(&config_dir.join("lumen_config.schema.json"))?;
    export::<lumen_schema::ModelInfo>(&model_dir.join("model_info.schema.json"))?;

    Ok(())
}

fn export<T>(path: &Path) -> Result<(), Box<dyn std::error::Error>>
where
    T: JsonSchema,
{
    let schema = schema_for!(T);
    let json = serde_json::to_string_pretty(&schema)?;
    fs::write(path, json)?;
    Ok(())
}

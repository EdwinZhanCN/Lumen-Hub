use std::{env, sync::OnceLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Language {
    En,
    ZhCn,
}

pub(crate) static LANGUAGE: OnceLock<Language> = OnceLock::new();

pub(crate) fn language() -> Language {
    *LANGUAGE.get_or_init(Language::detect)
}

impl Language {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::ZhCn => "zh-CN",
        }
    }

    fn detect() -> Self {
        ["LC_ALL", "LC_MESSAGES", "LANG"]
            .into_iter()
            .find_map(|name| {
                env::var(name)
                    .ok()
                    .and_then(|value| Self::parse_locale(&value))
            })
            .unwrap_or(Self::En)
    }

    fn parse_locale(value: &str) -> Option<Self> {
        let normalized = value
            .trim()
            .split(['.', '@'])
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .replace('_', "-");
        if normalized == "en" || normalized.starts_with("en-") {
            return Some(Self::En);
        }
        if matches!(normalized.as_str(), "zh" | "zh-cn" | "zh-hans")
            || normalized.starts_with("zh-cn-")
            || normalized.starts_with("zh-hans-")
        {
            return Some(Self::ZhCn);
        }
        None
    }

    fn parse_explicit(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "en" => Some(Self::En),
            "zh-cn" => Some(Self::ZhCn),
            _ => None,
        }
    }
}

pub(crate) fn extract_language(mut args: Vec<String>) -> Result<(Language, Vec<String>), String> {
    let mut selected = None;
    let mut index = 1;
    while index < args.len() {
        if args[index] == "--lang" {
            let value = args
                .get(index + 1)
                .ok_or_else(|| "missing value for `--lang`".to_owned())?
                .clone();
            selected = Language::parse_explicit(&value);
            if selected.is_none() {
                return Err(format!(
                    "unsupported language `{value}`; use `en` or `zh-CN`"
                ));
            }
            args.drain(index..=index + 1);
            continue;
        }
        if let Some(value) = args[index].strip_prefix("--lang=") {
            selected = Language::parse_explicit(value);
            if selected.is_none() {
                return Err(format!(
                    "unsupported language `{value}`; use `en` or `zh-CN`"
                ));
            }
            args.remove(index);
            continue;
        }
        index += 1;
    }
    Ok((selected.unwrap_or_else(Language::detect), args))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_flag_is_removed_before_command_parsing() {
        let args = vec![
            "lumen-cli".to_owned(),
            "--lang=zh-CN".to_owned(),
            "validate".to_owned(),
        ];
        let (language, args) = extract_language(args).unwrap();
        assert_eq!(language, Language::ZhCn);
        assert_eq!(args, vec!["lumen-cli", "validate"]);
    }

    #[test]
    fn language_parser_accepts_locale_forms_and_rejects_unknown_languages() {
        assert_eq!(Language::parse_locale("zh_CN.UTF-8"), Some(Language::ZhCn));
        assert_eq!(Language::parse_locale("zh-Hans-CN"), Some(Language::ZhCn));
        assert_eq!(Language::parse_locale("zh-TW"), None);
        assert_eq!(Language::parse_locale("en-US"), Some(Language::En));
        assert_eq!(Language::parse_locale("fr-FR"), None);
        assert_eq!(Language::parse_explicit("zh-CN"), Some(Language::ZhCn));
        assert_eq!(Language::parse_explicit("zh-TW"), None);
        assert_eq!(Language::parse_explicit("en-US"), None);
    }
}

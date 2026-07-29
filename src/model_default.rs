use std::{
    fs, io,
    path::{Component, Path, PathBuf},
};

/// A small, human-readable preference file kept beside the downloaded models.
/// Keeping it in `models_dir` makes the choice follow the desktop installation's
/// model library without introducing another platform-specific config path.
pub const DEFAULT_MODEL_PREFERENCE_FILE: &str = ".camelid-default-model";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefaultModelChoice {
    pub filename: String,
    pub path: PathBuf,
    /// `true` when the user explicitly chose this model. When false, Camelid is
    /// using the first local GGUF as a zero-configuration default.
    pub configured: bool,
}

pub fn valid_local_model_filename(filename: &str) -> bool {
    let path = Path::new(filename);
    filename == filename.trim()
        && !filename.is_empty()
        && !filename.contains(['/', '\\', ':'])
        && matches!(
            (path.components().next(), path.components().nth(1)),
            (Some(Component::Normal(_)), None)
        )
        && path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
}

fn existing_model(
    models_dir: &Path,
    filename: &str,
    configured: bool,
) -> Option<DefaultModelChoice> {
    if !valid_local_model_filename(filename) {
        return None;
    }
    let path = models_dir.join(filename);
    path.is_file().then(|| DefaultModelChoice {
        filename: filename.to_string(),
        path,
        configured,
    })
}

pub fn configured_default_model(models_dir: &Path) -> Option<DefaultModelChoice> {
    let filename = fs::read_to_string(models_dir.join(DEFAULT_MODEL_PREFERENCE_FILE)).ok()?;
    existing_model(models_dir, filename.trim(), true)
}

pub fn first_local_model(models_dir: &Path) -> Option<DefaultModelChoice> {
    let mut paths = fs::read_dir(models_dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    let path = paths.into_iter().next()?;
    let filename = path.file_name()?.to_str()?.to_string();
    Some(DefaultModelChoice {
        filename,
        path,
        configured: false,
    })
}

pub fn effective_default_model(models_dir: &Path) -> Option<DefaultModelChoice> {
    configured_default_model(models_dir).or_else(|| first_local_model(models_dir))
}

pub fn set_default_model(models_dir: &Path, filename: &str) -> io::Result<DefaultModelChoice> {
    if !valid_local_model_filename(filename) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "default model must be one local GGUF filename",
        ));
    }
    let Some(choice) = existing_model(models_dir, filename, true) else {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "default model does not exist in the configured models directory",
        ));
    };
    fs::write(
        models_dir.join(DEFAULT_MODEL_PREFERENCE_FILE),
        format!("{filename}\n"),
    )?;
    Ok(choice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_local_gguf_is_the_zero_configuration_default() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("zulu.gguf"), []).unwrap();
        fs::write(dir.path().join("alpha.GGUF"), []).unwrap();
        fs::write(dir.path().join("ignore.txt"), []).unwrap();

        let choice = effective_default_model(dir.path()).unwrap();
        assert_eq!(choice.filename, "alpha.GGUF");
        assert!(!choice.configured);
    }

    #[test]
    fn configured_default_wins_and_survives_a_rescan() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("alpha.gguf"), []).unwrap();
        fs::write(dir.path().join("chosen.gguf"), []).unwrap();

        let saved = set_default_model(dir.path(), "chosen.gguf").unwrap();
        assert_eq!(saved.filename, "chosen.gguf");
        assert!(saved.configured);

        let choice = effective_default_model(dir.path()).unwrap();
        assert_eq!(choice, saved);
    }

    #[test]
    fn stale_or_unsafe_preference_falls_back_to_first_local_model() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("available.gguf"), []).unwrap();
        fs::write(
            dir.path().join(DEFAULT_MODEL_PREFERENCE_FILE),
            "../outside.gguf\n",
        )
        .unwrap();

        let choice = effective_default_model(dir.path()).unwrap();
        assert_eq!(choice.filename, "available.gguf");
        assert!(!choice.configured);

        assert_eq!(
            set_default_model(dir.path(), "../outside.gguf")
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            set_default_model(dir.path(), "missing.gguf")
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
    }
}

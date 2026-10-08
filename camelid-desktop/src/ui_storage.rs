use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use tauri::{Manager, State};

const UI_STORAGE_FILE: &str = "ui-storage-v1.json";
const UI_STORAGE_TEMP_FILE: &str = "ui-storage-v1.json.tmp";
const UI_STORAGE_LOCK_FILE: &str = "ui-storage-v1.lock";
const UI_STORAGE_VERSION: u32 = 1;
const MAX_KEY_BYTES: usize = 256;
const MAX_VALUE_BYTES: usize = 32 * 1024 * 1024;
const MAX_DOCUMENT_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Default)]
struct UiStorageInner {
    initialized: bool,
    values: BTreeMap<String, String>,
}

#[derive(Debug, Default)]
pub struct UiStorageState(Mutex<UiStorageInner>);

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct UiStorageDocument {
    version: u32,
    values: BTreeMap<String, String>,
}

#[derive(serde::Serialize)]
struct UiStorageDocumentRef<'a> {
    version: u32,
    values: &'a BTreeMap<String, String>,
}

#[derive(Debug, serde::Serialize)]
pub struct UiStorageSnapshot {
    version: u32,
    initialized: bool,
    values: BTreeMap<String, String>,
}

fn storage_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(UI_STORAGE_FILE)
}

fn temp_storage_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(UI_STORAGE_TEMP_FILE)
}

fn validate_entry(key: &str, value: &str) -> Result<(), String> {
    if !key.starts_with("camelid") {
        return Err("desktop UI storage only accepts Camelid-owned keys".to_string());
    }
    if key.is_empty() || key.len() > MAX_KEY_BYTES {
        return Err(format!(
            "desktop UI storage keys must be between 1 and {MAX_KEY_BYTES} bytes"
        ));
    }
    if value.len() > MAX_VALUE_BYTES {
        return Err(format!(
            "desktop UI storage values may not exceed {MAX_VALUE_BYTES} bytes"
        ));
    }
    Ok(())
}

fn validate_values(values: &BTreeMap<String, String>) -> Result<(), String> {
    for (key, value) in values {
        validate_entry(key, value)?;
    }
    Ok(())
}

fn decode_document(bytes: &[u8], path: &Path) -> Result<UiStorageDocument, String> {
    let document: UiStorageDocument = serde_json::from_slice(bytes)
        .map_err(|err| format!("{} is invalid: {err}", path.display()))?;
    if document.version != UI_STORAGE_VERSION {
        return Err(format!(
            "{} uses unsupported UI storage version {}",
            path.display(),
            document.version
        ));
    }
    validate_values(&document.values)?;
    Ok(document)
}

fn load_document(app_data_dir: &Path) -> Result<(bool, BTreeMap<String, String>), String> {
    let path = storage_path(app_data_dir);
    match fs::read(&path) {
        Ok(bytes) => return decode_document(&bytes, &path).map(|document| (true, document.values)),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(format!("could not read {}: {err}", path.display())),
    }

    // On Windows replacing an existing file requires unlinking it first. If the
    // process stopped in that tiny interval, the complete, synced temporary file
    // is still the durable authority and can be recovered here.
    let temp_path = temp_storage_path(app_data_dir);
    match fs::read(&temp_path) {
        Ok(bytes) => {
            let document = decode_document(&bytes, &temp_path)?;
            replace_file(&temp_path, &path)?;
            Ok((true, document.values))
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok((false, BTreeMap::new())),
        Err(err) => Err(format!("could not read {}: {err}", temp_path.display())),
    }
}

fn replace_file(temp_path: &Path, path: &Path) -> Result<(), String> {
    #[cfg(windows)]
    if path.exists() {
        fs::remove_file(path)
            .map_err(|err| format!("could not replace {}: {err}", path.display()))?;
    }

    fs::rename(temp_path, path).map_err(|err| {
        format!(
            "could not move {} into {}: {err}",
            temp_path.display(),
            path.display()
        )
    })
}

fn write_document(app_data_dir: &Path, values: &BTreeMap<String, String>) -> Result<(), String> {
    validate_values(values)?;
    fs::create_dir_all(app_data_dir)
        .map_err(|err| format!("could not create {}: {err}", app_data_dir.display()))?;

    let bytes = serde_json::to_vec_pretty(&UiStorageDocumentRef {
        version: UI_STORAGE_VERSION,
        values,
    })
    .map_err(|err| format!("could not encode desktop UI storage: {err}"))?;
    if bytes.len() > MAX_DOCUMENT_BYTES {
        return Err(format!(
            "desktop UI storage may not exceed {MAX_DOCUMENT_BYTES} bytes"
        ));
    }

    let temp_path = temp_storage_path(app_data_dir);
    let path = storage_path(app_data_dir);
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temp_path)
        .map_err(|err| format!("could not create {}: {err}", temp_path.display()))?;
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .map_err(|err| format!("could not save {}: {err}", temp_path.display()))?;
    drop(file);
    replace_file(&temp_path, &path)
}

impl UiStorageState {
    fn with_document<T>(
        &self,
        app_data_dir: &Path,
        operation: impl FnOnce(&mut UiStorageInner) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut inner = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        fs::create_dir_all(app_data_dir)
            .map_err(|err| format!("could not create {}: {err}", app_data_dir.display()))?;
        // Lock a separate file: replacing the JSON must not replace the inode
        // whose lock protects the transaction. Dropping the file unlocks it.
        let lock_path = app_data_dir.join(UI_STORAGE_LOCK_FILE);
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|err| format!("could not open {}: {err}", lock_path.display()))?;
        lock.lock()
            .map_err(|err| format!("could not lock {}: {err}", lock_path.display()))?;
        // Every transaction starts from the durable document, including reads.
        // A stale state must never overwrite keys saved by another state.
        let (initialized, values) = load_document(app_data_dir)?;
        inner.initialized = initialized;
        inner.values = values;
        operation(&mut inner)
    }

    fn snapshot(&self, app_data_dir: &Path) -> Result<UiStorageSnapshot, String> {
        self.with_document(app_data_dir, |inner| {
            Ok(UiStorageSnapshot {
                version: UI_STORAGE_VERSION,
                initialized: inner.initialized,
                values: inner.values.clone(),
            })
        })
    }

    fn set_value(
        &self,
        app_data_dir: &Path,
        key: String,
        value: Option<String>,
    ) -> Result<(), String> {
        if let Some(value) = value.as_deref() {
            validate_entry(&key, value)?;
        } else if !key.starts_with("camelid") || key.len() > MAX_KEY_BYTES {
            return Err("desktop UI storage only accepts Camelid-owned keys".to_string());
        }

        self.with_document(app_data_dir, |inner| {
            let mut next = inner.values.clone();
            if let Some(value) = value {
                next.insert(key, value);
            } else {
                next.remove(&key);
            }
            write_document(app_data_dir, &next)?;
            inner.initialized = true;
            inner.values = next;
            Ok(())
        })
    }

    fn replace_values(
        &self,
        app_data_dir: &Path,
        values: BTreeMap<String, String>,
    ) -> Result<(), String> {
        validate_values(&values)?;
        self.with_document(app_data_dir, |inner| {
            write_document(app_data_dir, &values)?;
            inner.initialized = true;
            inner.values = values;
            Ok(())
        })
    }
}

fn app_data_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_data_dir()
        .map_err(|err| format!("could not resolve Camelid application data: {err}"))
}

#[tauri::command]
pub async fn read_ui_storage(
    app: tauri::AppHandle,
    state: State<'_, UiStorageState>,
) -> Result<UiStorageSnapshot, String> {
    state.snapshot(&app_data_dir(&app)?)
}

#[tauri::command]
pub async fn set_ui_storage_value(
    app: tauri::AppHandle,
    state: State<'_, UiStorageState>,
    key: String,
    value: Option<String>,
) -> Result<(), String> {
    state.set_value(&app_data_dir(&app)?, key, value)
}

#[tauri::command]
pub async fn replace_ui_storage(
    app: tauri::AppHandle,
    state: State<'_, UiStorageState>,
    values: BTreeMap<String, String>,
) -> Result<(), String> {
    state.replace_values(&app_data_dir(&app)?, values)
}

#[cfg(test)]
mod tests {
    use super::{storage_path, UiStorageState};
    use std::collections::BTreeMap;

    #[test]
    fn desktop_state_survives_a_fresh_webview_origin() {
        let root = tempfile::tempdir().unwrap();
        let first_launch = UiStorageState::default();
        first_launch
            .set_value(
                root.path(),
                "camelid-theme".to_string(),
                Some("light".to_string()),
            )
            .unwrap();
        first_launch
            .set_value(
                root.path(),
                "camelid.conversations".to_string(),
                Some("[{\"id\":\"chat-1\"}]".to_string()),
            )
            .unwrap();

        // A new state instance models a desktop restart whose WebView has a
        // different ephemeral-port origin and therefore a fresh localStorage.
        let second_launch = UiStorageState::default();
        let snapshot = second_launch.snapshot(root.path()).unwrap();
        assert!(snapshot.initialized);
        assert_eq!(snapshot.values["camelid-theme"], "light");
        assert_eq!(
            snapshot.values["camelid.conversations"],
            "[{\"id\":\"chat-1\"}]"
        );
    }

    #[test]
    fn initialized_empty_storage_does_not_resurrect_stale_origin_values() {
        let root = tempfile::tempdir().unwrap();
        let first_launch = UiStorageState::default();
        first_launch
            .replace_values(root.path(), BTreeMap::new())
            .unwrap();

        let second_launch = UiStorageState::default();
        let snapshot = second_launch.snapshot(root.path()).unwrap();
        assert!(snapshot.initialized);
        assert!(snapshot.values.is_empty());
        assert!(storage_path(root.path()).is_file());
    }

    #[test]
    fn desktop_storage_rejects_non_camelid_keys() {
        let root = tempfile::tempdir().unwrap();
        let state = UiStorageState::default();
        let error = state
            .set_value(
                root.path(),
                "unrelated-key".to_string(),
                Some("value".to_string()),
            )
            .unwrap_err();
        assert!(error.contains("Camelid-owned"));
        assert!(!storage_path(root.path()).exists());
    }

    #[test]
    fn unrelated_write_from_a_stale_state_preserves_saved_chats() {
        let root = tempfile::tempdir().unwrap();
        let first = UiStorageState::default();
        let second = UiStorageState::default();
        first.snapshot(root.path()).unwrap();
        second.snapshot(root.path()).unwrap();
        first
            .set_value(
                root.path(),
                "camelid.conversations".into(),
                Some("saved chat".into()),
            )
            .unwrap();
        second
            .set_value(root.path(), "camelid-theme".into(), Some("light".into()))
            .unwrap();
        let snapshot = first.snapshot(root.path()).unwrap();
        assert_eq!(snapshot.values["camelid.conversations"], "saved chat");
        assert_eq!(snapshot.values["camelid-theme"], "light");

        first
            .set_value(root.path(), "camelid-theme".into(), None)
            .unwrap();
        assert_eq!(second.snapshot(root.path()).unwrap().values.len(), 1);
    }

    #[test]
    fn concurrent_storage_transactions_preserve_every_independent_key() {
        let root = tempfile::tempdir().unwrap();
        std::thread::scope(|scope| {
            for index in 0..16 {
                let path = root.path();
                scope.spawn(move || {
                    UiStorageState::default()
                        .set_value(
                            path,
                            format!("camelid.test-{index}"),
                            Some(index.to_string()),
                        )
                        .unwrap();
                });
            }
        });
        let snapshot = UiStorageState::default().snapshot(root.path()).unwrap();
        for index in 0..16 {
            assert_eq!(
                snapshot.values[&format!("camelid.test-{index}")],
                index.to_string()
            );
        }
    }
}

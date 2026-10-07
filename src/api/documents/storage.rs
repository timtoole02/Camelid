//! Persistent Knowledge Library location and one-time legacy snapshot migration.
use std::fs::{self, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OpenFlags};

const DB_FILE: &str = "documents_rag.sqlite3";

struct Location {
    path: PathBuf,
    migrate_legacy: bool,
}

fn location(os: &str, env: impl Fn(&str) -> Option<PathBuf>) -> io::Result<Location> {
    let value = |key| env(key).filter(|path| !path.as_os_str().is_empty());
    if let Some(dir) = value("CAMELID_DATA_DIR") {
        return Ok(Location {
            path: dir.join(DB_FILE),
            migrate_legacy: false,
        });
    }
    let dir = match os {
        "windows" => value("LOCALAPPDATA")
            .map(|dir| dir.join("Camelid"))
            .or_else(|| value("USERPROFILE").map(|dir| dir.join("AppData/Local/Camelid"))),
        "macos" => value("HOME").map(|dir| dir.join("Library/Application Support/Camelid")),
        _ => value("XDG_DATA_HOME")
            .map(|dir| dir.join("camelid"))
            .or_else(|| value("HOME").map(|dir| dir.join(".local/share/camelid"))),
    };
    let dir = dir.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "could not locate persistent application data; set CAMELID_DATA_DIR",
        )
    })?;
    Ok(Location {
        path: dir.join(DB_FILE),
        migrate_legacy: true,
    })
}

fn current_location() -> io::Result<Location> {
    location(std::env::consts::OS, |key| {
        std::env::var_os(key).map(PathBuf::from)
    })
}

fn io_error(error: io::Error) -> rusqlite::Error {
    rusqlite::Error::SqliteFailure(
        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CANTOPEN),
        Some(error.to_string()),
    )
}

fn eligible_legacy(path: &Path) -> bool {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return false;
    };
    if !metadata.file_type().is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        // The old Linux filename lived in a shared temporary directory.
        if metadata.uid() != unsafe { libc::geteuid() } {
            return false;
        }
    }
    true
}

pub(super) fn library_exists() -> bool {
    current_location().is_ok_and(|location| {
        location.path.exists()
            || (location.migrate_legacy && eligible_legacy(&std::env::temp_dir().join(DB_FILE)))
    })
}

pub(super) fn prepare_path() -> rusqlite::Result<PathBuf> {
    let location = current_location().map_err(io_error)?;
    let parent = location
        .path
        .parent()
        .ok_or_else(|| rusqlite::Error::InvalidPath(location.path.clone()))?;
    fs::create_dir_all(parent).map_err(io_error)?;
    #[cfg(unix)]
    if location.migrate_legacy {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(parent, fs::Permissions::from_mode(0o700)).map_err(io_error)?;
    }
    if location.migrate_legacy {
        migrate(&std::env::temp_dir().join(DB_FILE), &location.path)?;
    }
    Ok(location.path)
}

fn migrate(legacy: &Path, destination: &Path) -> rusqlite::Result<()> {
    if destination.exists() || !eligible_legacy(legacy) {
        return Ok(());
    }
    let parent = destination
        .parent()
        .ok_or_else(|| rusqlite::Error::InvalidPath(destination.into()))?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(parent.join("documents-migration.lock"))
        .map_err(io_error)?;
    lock.lock().map_err(io_error)?;
    if destination.exists() || !eligible_legacy(legacy) {
        return Ok(());
    }
    // macOS aliases /var to /private/var. Resolve directory aliases while
    // leaving the database filename unresolved so NOFOLLOW still guards it.
    let source_path = legacy
        .parent()
        .ok_or_else(|| rusqlite::Error::InvalidPath(legacy.into()))?
        .canonicalize()
        .map_err(io_error)?
        .join(
            legacy
                .file_name()
                .ok_or_else(|| rusqlite::Error::InvalidPath(legacy.into()))?,
        );
    if !eligible_legacy(&source_path) {
        return Ok(());
    }
    let source = Connection::open_with_flags(
        source_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NOFOLLOW,
    )?;
    source.busy_timeout(std::time::Duration::from_secs(5))?;
    // SQLite takes a consistent snapshot, including committed WAL content.
    // Publish only a complete snapshot; never replace an existing library.
    // VACUUM inherits NOFOLLOW for its output, so its directory must also
    // resolve system aliases such as macOS /var before SQLite opens it.
    let snapshot_dir = parent.canonicalize().map_err(io_error)?;
    let snapshot = tempfile::NamedTempFile::new_in(snapshot_dir).map_err(io_error)?;
    let snapshot_path = snapshot
        .path()
        .to_str()
        .ok_or_else(|| rusqlite::Error::InvalidPath(snapshot.path().into()))?;
    source.execute("VACUUM INTO ?1", [snapshot_path])?;
    snapshot.as_file().sync_all().map_err(io_error)?;
    snapshot
        .persist_noclobber(destination)
        .map_err(|error| io_error(error.error))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_persistent_per_user_on_every_supported_platform() {
        let root = tempfile::tempdir().unwrap();
        for (os, key, suffix) in [
            ("windows", "LOCALAPPDATA", "Camelid"),
            ("windows", "USERPROFILE", "AppData/Local/Camelid"),
            ("macos", "HOME", "Library/Application Support/Camelid"),
            ("linux", "XDG_DATA_HOME", "camelid"),
            ("linux", "HOME", ".local/share/camelid"),
        ] {
            let resolved = location(os, |name| (name == key).then(|| root.path().into())).unwrap();
            assert_eq!(resolved.path, root.path().join(suffix).join(DB_FILE));
            assert!(resolved.migrate_legacy);
        }
        assert!(
            location("linux", |_| None).is_err(),
            "never silently return to temporary storage"
        );
        let explicit = location("linux", |name| {
            (name == "CAMELID_DATA_DIR").then(|| root.path().into())
        })
        .unwrap();
        assert_eq!(explicit.path, root.path().join(DB_FILE));
        assert!(!explicit.migrate_legacy);
    }

    #[test]
    fn migration_preserves_documents_collections_folders_and_wal_content() {
        let root = tempfile::tempdir().unwrap();
        let legacy = root.path().join("legacy.sqlite3");
        let destination = root.path().join(DB_FILE);
        let source = Connection::open(&legacy).unwrap();
        source
            .execute_batch("PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0;")
            .unwrap();
        super::super::init_db(&source).unwrap();
        source.execute_batch(
            "INSERT INTO documents (id,filename,file_type,byte_size,chunk_count,created_at) VALUES ('doc','notes.txt','txt',5,1,1);
             INSERT INTO document_chunks (id,doc_id,chunk_index,content) VALUES (17,'doc',0,'hello');
             INSERT INTO document_chunks_fts (rowid,content) VALUES (17,'hello');
             INSERT INTO document_collections VALUES ('collection','Notes',1);
             INSERT INTO document_collection_members VALUES ('collection','doc',1);
             INSERT INTO watched_folders (id,path,collection_id,created_at) VALUES ('folder','notes','collection',1);"
        ).unwrap();
        migrate(&legacy, &destination).unwrap();
        drop(source);
        fs::remove_file(&legacy).unwrap();
        let saved = Connection::open(&destination).unwrap();
        for query in [
            "SELECT COUNT(*) FROM documents WHERE id='doc'",
            "SELECT COUNT(*) FROM document_collection_members WHERE doc_id='doc'",
            "SELECT COUNT(*) FROM watched_folders WHERE id='folder'",
            "SELECT COUNT(*) FROM document_chunks_fts WHERE document_chunks_fts MATCH 'hello' AND rowid=17",
        ] {
            assert_eq!(saved.query_row(query, [], |row| row.get::<_, i64>(0)).unwrap(), 1);
        }
    }

    #[test]
    fn migration_never_overwrites_a_current_library_or_publishes_a_partial_copy() {
        let root = tempfile::tempdir().unwrap();
        let legacy = root.path().join("legacy.sqlite3");
        let destination = root.path().join(DB_FILE);
        fs::write(&legacy, b"not a sqlite database").unwrap();
        assert!(migrate(&legacy, &destination).is_err());
        assert!(!destination.exists());
        fs::write(&destination, b"current library").unwrap();
        migrate(&legacy, &destination).unwrap();
        assert_eq!(fs::read(&destination).unwrap(), b"current library");
        assert!(legacy.exists());
    }

    #[cfg(unix)]
    #[test]
    fn migration_accepts_system_directory_aliases() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().canonicalize().unwrap().join("temporary");
        fs::create_dir(&target).unwrap();
        let alias = root.path().join("alias");
        std::os::unix::fs::symlink(&target, &alias).unwrap();
        let source = Connection::open(target.join("legacy.sqlite3")).unwrap();
        source
            .execute_batch("CREATE TABLE notes (text TEXT); INSERT INTO notes VALUES ('saved');")
            .unwrap();
        let destination = alias.join(DB_FILE);
        migrate(&alias.join("legacy.sqlite3"), &destination).unwrap();
        let saved = Connection::open(destination).unwrap();
        assert_eq!(
            saved
                .query_row("SELECT text FROM notes", [], |row| row.get::<_, String>(0))
                .unwrap(),
            "saved"
        );
    }

    #[cfg(unix)]
    #[test]
    fn migration_does_not_follow_a_legacy_symlink() {
        let root = tempfile::tempdir().unwrap();
        let legacy = root.path().join("legacy.sqlite3");
        std::os::unix::fs::symlink(root.path().join("other-user-data"), &legacy).unwrap();
        let destination = root.path().join(DB_FILE);
        migrate(&legacy, &destination).unwrap();
        assert!(!destination.exists());
    }
}

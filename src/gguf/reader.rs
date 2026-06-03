use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{ErrorKind, Read},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{BackendError, Result};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
const GGML_MAX_DIMS: u32 = 4;
const GGML_MAX_NAME: usize = 64;

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
pub enum GgufMetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufMetadataValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum GgufTensorType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ4NL,
    I8,
    I16,
    I32,
    I64,
    F64,
    BF16,
    Unknown(i32),
}

impl GgufTensorType {
    pub fn from_id(value: i32) -> Self {
        match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            20 => Self::IQ4NL,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::BF16,
            other => Self::Unknown(other),
        }
    }

    pub fn layout(self) -> Option<(u64, u64)> {
        // (block_size, type_size_bytes), matching common GGML storage sizes.
        match self {
            Self::F32 => Some((1, 4)),
            Self::F16 => Some((1, 2)),
            Self::Q4_0 | Self::Q4_1 => Some((32, 18)),
            Self::Q5_0 | Self::Q5_1 => Some((32, 22)),
            Self::Q8_0 => Some((32, 34)),
            Self::Q8_1 => Some((32, 36)),
            Self::Q2K => Some((256, 84)),
            Self::Q3K => Some((256, 110)),
            Self::Q4K => Some((256, 144)),
            Self::Q5K => Some((256, 176)),
            Self::Q6K => Some((256, 210)),
            Self::Q8K => Some((256, 292)),
            Self::IQ4NL => Some((32, 18)),
            Self::I8 => Some((1, 1)),
            Self::I16 | Self::BF16 => Some((1, 2)),
            Self::I32 => Some((1, 4)),
            Self::I64 | Self::F64 => Some((1, 8)),
            Self::Unknown(_) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GgufTensorDescriptor {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: GgufTensorType,
    pub relative_offset: u64,
    pub absolute_offset: u64,
    pub n_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct GgufFile {
    pub path: PathBuf,
    pub version: u32,
    pub tensor_count: i64,
    pub metadata_count: i64,
    pub alignment: u64,
    pub data_start_offset: u64,
    pub metadata: BTreeMap<String, GgufMetadataValue>,
    pub tensors: Vec<GgufTensorDescriptor>,
}

impl GgufFile {
    pub fn architecture(&self) -> Option<&str> {
        self.metadata_string("general.architecture")
    }

    pub fn model_name(&self) -> Option<&str> {
        self.metadata_string("general.name")
    }

    pub fn metadata_string(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(GgufMetadataValue::String(value)) => Some(value.as_str()),
            _ => None,
        }
    }

    pub fn metadata_bool(&self, key: &str) -> Option<bool> {
        match self.metadata.get(key) {
            Some(GgufMetadataValue::Bool(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(GgufMetadataValue::U32(value)) => Some(*value),
            Some(GgufMetadataValue::U64(value)) => (*value).try_into().ok(),
            _ => None,
        }
    }

    pub fn metadata_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key) {
            Some(GgufMetadataValue::F32(value)) => Some(*value),
            Some(GgufMetadataValue::F64(value)) => Some(*value as f32),
            _ => None,
        }
    }

    pub fn metadata_array_strings(&self, key: &str) -> Result<Vec<String>> {
        match self.metadata.get(key) {
            Some(GgufMetadataValue::Array(values)) => values
                .iter()
                .map(|value| match value {
                    GgufMetadataValue::String(value) => Ok(value.clone()),
                    _ => Err(BackendError::InvalidGguf(format!(
                        "metadata {key} must be array<string>"
                    ))),
                })
                .collect(),
            Some(_) => Err(BackendError::InvalidGguf(format!(
                "metadata {key} must be array<string>"
            ))),
            None => Err(BackendError::InvalidGguf(format!(
                "required metadata {key} is missing"
            ))),
        }
    }

    pub fn metadata_array_strings_optional(&self, key: &str) -> Result<Option<Vec<String>>> {
        match self.metadata.get(key) {
            Some(GgufMetadataValue::Array(values)) => values
                .iter()
                .map(|value| match value {
                    GgufMetadataValue::String(value) => Ok(value.clone()),
                    _ => Err(BackendError::InvalidGguf(format!(
                        "metadata {key} must be array<string>"
                    ))),
                })
                .collect::<Result<Vec<_>>>()
                .map(Some),
            Some(_) => Err(BackendError::InvalidGguf(format!(
                "metadata {key} must be array<string>"
            ))),
            None => Ok(None),
        }
    }

    pub fn metadata_array_f32_optional(&self, key: &str) -> Result<Option<Vec<f32>>> {
        match self.metadata.get(key) {
            Some(GgufMetadataValue::Array(values)) => values
                .iter()
                .map(|value| match value {
                    GgufMetadataValue::F32(value) => Ok(*value),
                    _ => Err(BackendError::InvalidGguf(format!(
                        "metadata {key} must be array<float>"
                    ))),
                })
                .collect::<Result<Vec<_>>>()
                .map(Some),
            Some(_) => Err(BackendError::InvalidGguf(format!(
                "metadata {key} must be array<float>"
            ))),
            None => Ok(None),
        }
    }

    pub fn metadata_array_i32_optional(&self, key: &str) -> Result<Option<Vec<i32>>> {
        match self.metadata.get(key) {
            Some(GgufMetadataValue::Array(values)) => values
                .iter()
                .map(|value| match value {
                    GgufMetadataValue::I32(value) => Ok(*value),
                    GgufMetadataValue::U32(value) => (*value).try_into().map_err(|_| {
                        BackendError::InvalidGguf(format!(
                            "metadata {key} contains u32 too large for i32"
                        ))
                    }),
                    _ => Err(BackendError::InvalidGguf(format!(
                        "metadata {key} must be array<int>"
                    ))),
                })
                .collect::<Result<Vec<_>>>()
                .map(Some),
            Some(_) => Err(BackendError::InvalidGguf(format!(
                "metadata {key} must be array<int>"
            ))),
            None => Ok(None),
        }
    }
}

pub fn read_metadata(path: &Path) -> Result<GgufFile> {
    let file = File::open(path).map_err(|source| BackendError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let file_len = fs::metadata(path)
        .map_err(|source| BackendError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    let mut cursor = Cursor::new(file, path.to_path_buf());

    let magic = cursor.read_bytes(4)?;
    if magic != GGUF_MAGIC {
        return Err(BackendError::InvalidGguf(
            "bad magic; expected GGUF".to_string(),
        ));
    }

    let version = cursor.read_u32()?;
    if !(2..=3).contains(&version) {
        return Err(BackendError::UnsupportedGguf(format!(
            "version {version}; expected v2 or v3"
        )));
    }

    let tensor_count = cursor.read_i64()?;
    let metadata_count = cursor.read_i64()?;
    if tensor_count < 0 || metadata_count < 0 {
        return Err(BackendError::InvalidGguf(
            "negative tensor or metadata count".to_string(),
        ));
    }

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = cursor.read_string()?;
        let value = read_value(&mut cursor)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(BackendError::InvalidGguf(format!(
                "duplicate metadata key {key}"
            )));
        }
    }

    let alignment = match metadata.get("general.alignment") {
        Some(GgufMetadataValue::U32(value)) => u64::from(*value),
        Some(GgufMetadataValue::U64(value)) => *value,
        Some(_) => {
            return Err(BackendError::InvalidGguf(
                "general.alignment has non-integer type".to_string(),
            ))
        }
        None => DEFAULT_ALIGNMENT,
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(BackendError::InvalidGguf(format!(
            "invalid alignment {alignment}"
        )));
    }

    let mut raw_tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = cursor.read_string()?;
        if name.len() >= GGML_MAX_NAME {
            return Err(BackendError::InvalidGguf(format!(
                "tensor name {name} exceeds GGML_MAX_NAME"
            )));
        }
        let n_dimensions = cursor.read_u32()?;
        if n_dimensions == 0 || n_dimensions > GGML_MAX_DIMS {
            return Err(BackendError::InvalidGguf(format!(
                "tensor {name} has invalid dimension count {n_dimensions}"
            )));
        }
        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            let dim = cursor.read_i64()?;
            if dim < 0 {
                return Err(BackendError::InvalidGguf(format!(
                    "tensor {name} has negative dimension {dim}"
                )));
            }
            dimensions.push(dim as u64);
        }
        let tensor_type = GgufTensorType::from_id(cursor.read_i32()?);
        let relative_offset = cursor.read_u64()?;
        raw_tensors.push((name, dimensions, tensor_type, relative_offset));
    }

    let data_start_offset = align_to(cursor.position(), alignment)?;
    if data_start_offset > file_len {
        return Err(BackendError::InvalidGguf(
            "aligned tensor data start is beyond end of file".to_string(),
        ));
    }

    let mut tensors = Vec::with_capacity(raw_tensors.len());
    let mut seen_tensor_names = std::collections::BTreeSet::new();
    let mut expected_offset = 0u64;
    for (name, dimensions, tensor_type, relative_offset) in raw_tensors {
        if !seen_tensor_names.insert(name.clone()) {
            return Err(BackendError::InvalidGguf(format!(
                "duplicate tensor name {name}"
            )));
        }
        if relative_offset != expected_offset {
            return Err(BackendError::InvalidGguf(format!("tensor {name} offset {relative_offset} is not contiguous; expected {expected_offset}")));
        }
        let n_bytes = tensor_nbytes(&name, &dimensions, tensor_type)?;
        let absolute_offset = data_start_offset
            .checked_add(relative_offset)
            .ok_or_else(|| {
                BackendError::InvalidGguf(format!("tensor {name} absolute offset overflow"))
            })?;
        let end = absolute_offset.checked_add(n_bytes).ok_or_else(|| {
            BackendError::InvalidGguf(format!("tensor {name} byte range overflow"))
        })?;
        if end > file_len {
            return Err(BackendError::InvalidGguf(format!(
                "tensor {name} data extends beyond end of file"
            )));
        }
        tensors.push(GgufTensorDescriptor {
            name,
            dimensions,
            tensor_type,
            relative_offset,
            absolute_offset,
            n_bytes,
        });
        expected_offset = align_to(
            relative_offset
                .checked_add(n_bytes)
                .ok_or_else(|| BackendError::InvalidGguf("tensor offset overflow".to_string()))?,
            alignment,
        )?;
    }

    Ok(GgufFile {
        path: path.to_path_buf(),
        version,
        tensor_count,
        metadata_count,
        alignment,
        data_start_offset,
        metadata,
        tensors,
    })
}

fn read_value(cursor: &mut Cursor) -> Result<GgufMetadataValue> {
    let ty = cursor.read_i32()?;
    read_value_of_type(cursor, ty)
}

fn read_value_of_type(cursor: &mut Cursor, ty: i32) -> Result<GgufMetadataValue> {
    Ok(match ty {
        0 => GgufMetadataValue::U8(cursor.read_u8()?),
        1 => GgufMetadataValue::I8(cursor.read_i8()?),
        2 => GgufMetadataValue::U16(cursor.read_u16()?),
        3 => GgufMetadataValue::I16(cursor.read_i16()?),
        4 => GgufMetadataValue::U32(cursor.read_u32()?),
        5 => GgufMetadataValue::I32(cursor.read_i32()?),
        6 => GgufMetadataValue::F32(cursor.read_f32()?),
        7 => GgufMetadataValue::Bool(cursor.read_bool()?),
        8 => GgufMetadataValue::String(cursor.read_string()?),
        9 => {
            let element_ty = cursor.read_i32()?;
            if element_ty == 9 {
                return Err(BackendError::UnsupportedGguf(
                    "nested metadata arrays".to_string(),
                ));
            }
            let len = cursor.read_u64()?;
            if len > 1_000_000 {
                return Err(BackendError::InvalidGguf(format!(
                    "metadata array too large: {len}"
                )));
            }
            let mut values = Vec::with_capacity(len as usize);
            for _ in 0..len {
                values.push(read_value_of_type(cursor, element_ty)?);
            }
            GgufMetadataValue::Array(values)
        }
        10 => GgufMetadataValue::U64(cursor.read_u64()?),
        11 => GgufMetadataValue::I64(cursor.read_i64()?),
        12 => GgufMetadataValue::F64(cursor.read_f64()?),
        other => {
            return Err(BackendError::UnsupportedGguf(format!(
                "metadata value type {other}"
            )))
        }
    })
}

fn tensor_nbytes(name: &str, dimensions: &[u64], tensor_type: GgufTensorType) -> Result<u64> {
    let (block_size, type_size) = tensor_type.layout().ok_or_else(|| {
        BackendError::UnsupportedGguf(format!(
            "tensor {name} has unknown or removed GGML type {tensor_type:?}"
        ))
    })?;
    let first_dim = *dimensions.first().unwrap_or(&1);
    if !first_dim.is_multiple_of(block_size) {
        return Err(BackendError::InvalidGguf(format!(
            "tensor {name} first dimension {first_dim} is not divisible by block size {block_size}"
        )));
    }
    let mut elements = 1u64;
    for dim in dimensions {
        elements = elements.checked_mul(*dim).ok_or_else(|| {
            BackendError::InvalidGguf(format!("tensor {name} element count overflow"))
        })?;
    }
    elements
        .checked_div(block_size)
        .and_then(|blocks| blocks.checked_mul(type_size))
        .ok_or_else(|| BackendError::InvalidGguf(format!("tensor {name} byte size overflow")))
}

fn align_to(value: u64, alignment: u64) -> Result<u64> {
    let add = alignment - 1;
    value
        .checked_add(add)
        .map(|v| v & !add)
        .ok_or_else(|| BackendError::InvalidGguf("alignment overflow".to_string()))
}

struct Cursor {
    reader: File,
    path: PathBuf,
    pos: u64,
}

impl Cursor {
    fn new(reader: File, path: PathBuf) -> Self {
        Self {
            reader,
            path,
            pos: 0,
        }
    }
    fn position(&self) -> u64 {
        self.pos
    }

    fn read_exact_into(&mut self, out: &mut [u8]) -> Result<()> {
        self.reader.read_exact(out).map_err(|source| {
            if source.kind() == ErrorKind::UnexpectedEof {
                BackendError::InvalidGguf("unexpected end of file".to_string())
            } else {
                BackendError::Io {
                    path: self.path.clone(),
                    source,
                }
            }
        })?;
        self.pos = self
            .pos
            .checked_add(out.len() as u64)
            .ok_or_else(|| BackendError::InvalidGguf("cursor overflow".to_string()))?;
        Ok(())
    }

    fn read_bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut out = vec![0; n];
        self.read_exact_into(&mut out)?;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8> {
        let mut b = [0; 1];
        self.read_exact_into(&mut b)?;
        Ok(b[0])
    }
    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }
    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }
    fn read_u16(&mut self) -> Result<u16> {
        let mut b = [0; 2];
        self.read_exact_into(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn read_i16(&mut self) -> Result<i16> {
        let mut b = [0; 2];
        self.read_exact_into(&mut b)?;
        Ok(i16::from_le_bytes(b))
    }
    fn read_u32(&mut self) -> Result<u32> {
        let mut b = [0; 4];
        self.read_exact_into(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn read_i32(&mut self) -> Result<i32> {
        let mut b = [0; 4];
        self.read_exact_into(&mut b)?;
        Ok(i32::from_le_bytes(b))
    }
    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }
    fn read_u64(&mut self) -> Result<u64> {
        let mut b = [0; 8];
        self.read_exact_into(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn read_i64(&mut self) -> Result<i64> {
        let mut b = [0; 8];
        self.read_exact_into(&mut b)?;
        Ok(i64::from_le_bytes(b))
    }
    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()?;
        if len > 16 * 1024 * 1024 {
            return Err(BackendError::InvalidGguf(format!(
                "string too large: {len}"
            )));
        }
        let bytes = self.read_bytes(len as usize)?;
        String::from_utf8(bytes)
            .map_err(|_| BackendError::InvalidGguf("invalid UTF-8 string".to_string()))
    }
}

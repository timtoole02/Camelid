use std::io::{self, Read, Write};

pub const ACTIVATION_MAGIC: u32 = 0xDEAD_BEEF;
pub const TOKEN_FEEDBACK_MAGIC: u32 = 0xCAFE_FEED;
pub const ACTIVATION_HEADER_BYTES: usize = 16;
pub const TOKEN_FEEDBACK_BYTES: usize = 12;
pub const DEFAULT_MAX_ACTIVATION_FLOATS: usize = 1 << 22; // up to 16MB of floats for 405B models

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActivationHeader {
    pub pos: u32,
    pub seq_len: u32,
    pub float_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenFeedback {
    pub token_id: u32,
    pub is_finished: bool,
}

pub fn send_activation_packet<W: Write>(
    writer: &mut W,
    pos: u32,
    seq_len: u32,
    activations: &[f32],
) -> io::Result<()> {
    let float_count = u32::try_from(activations.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "activation vector has too many floats: {}",
                activations.len()
            ),
        )
    })?;

    let mut header = [0_u8; ACTIVATION_HEADER_BYTES];
    header[0..4].copy_from_slice(&ACTIVATION_MAGIC.to_le_bytes());
    header[4..8].copy_from_slice(&pos.to_le_bytes());
    header[8..12].copy_from_slice(&seq_len.to_le_bytes());
    header[12..16].copy_from_slice(&float_count.to_le_bytes());
    writer.write_all(&header)?;

    // On little-endian architectures (Apple Silicon / Intel Macs), floats are already in native little-endian.
    // Interpret the float slice as a byte slice directly to write it in bulk with zero allocations and zero looping.
    let payload = unsafe {
        std::slice::from_raw_parts(
            activations.as_ptr() as *const u8,
            std::mem::size_of_val(activations),
        )
    };
    writer.write_all(payload)?;
    writer.flush()
}

pub fn recv_activation_packet<R: Read>(
    reader: &mut R,
    out_activations: &mut Vec<f32>,
) -> io::Result<ActivationHeader> {
    recv_activation_packet_limited(reader, out_activations, DEFAULT_MAX_ACTIVATION_FLOATS)
}

pub fn recv_activation_packet_limited<R: Read>(
    reader: &mut R,
    out_activations: &mut Vec<f32>,
    max_float_count: usize,
) -> io::Result<ActivationHeader> {
    let mut header = [0_u8; ACTIVATION_HEADER_BYTES];
    reader.read_exact(&mut header)?;

    let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
    if magic != ACTIVATION_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid activation header magic: 0x{magic:08X}"),
        ));
    }

    let pos = u32::from_le_bytes(header[4..8].try_into().unwrap());
    let seq_len = u32::from_le_bytes(header[8..12].try_into().unwrap());
    let float_count = u32::from_le_bytes(header[12..16].try_into().unwrap());
    let float_count_usize = float_count as usize;
    if float_count_usize > max_float_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("activation float_count {float_count_usize} exceeds limit {max_float_count}"),
        ));
    }

    // Resize the existing buffer. This eliminates all temporary byte buffer allocations.
    out_activations.resize(float_count_usize, 0.0);

    // Interpret the float vector buffer as a mutable byte buffer and read directly into it in one system call.
    let byte_count = float_count_usize * std::mem::size_of::<f32>();
    let payload_slice = unsafe {
        std::slice::from_raw_parts_mut(out_activations.as_mut_ptr() as *mut u8, byte_count)
    };
    reader.read_exact(payload_slice)?;

    Ok(ActivationHeader {
        pos,
        seq_len,
        float_count,
    })
}

pub fn send_token_feedback<W: Write>(
    writer: &mut W,
    token_id: u32,
    is_finished: bool,
) -> io::Result<()> {
    let mut bytes = [0_u8; TOKEN_FEEDBACK_BYTES];
    bytes[0..4].copy_from_slice(&TOKEN_FEEDBACK_MAGIC.to_le_bytes());
    bytes[4..8].copy_from_slice(&token_id.to_le_bytes());
    bytes[8..12].copy_from_slice(&(is_finished as u32).to_le_bytes());
    writer.write_all(&bytes)?;
    writer.flush()
}

pub fn recv_token_feedback<R: Read>(reader: &mut R) -> io::Result<TokenFeedback> {
    let mut bytes = [0_u8; TOKEN_FEEDBACK_BYTES];
    reader.read_exact(&mut bytes)?;

    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if magic != TOKEN_FEEDBACK_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid token feedback magic: 0x{magic:08X}"),
        ));
    }

    let token_id = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let is_finished = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    match is_finished {
        0 | 1 => Ok(TokenFeedback {
            token_id,
            is_finished: is_finished == 1,
        }),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid token feedback is_finished value: {other}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{
        recv_activation_packet, recv_activation_packet_limited, recv_token_feedback,
        send_activation_packet, send_token_feedback, ACTIVATION_MAGIC, TOKEN_FEEDBACK_MAGIC,
    };

    #[test]
    fn activation_packet_round_trips_little_endian_payload() {
        let activations = [0.0_f32, 1.25, -2.5, f32::INFINITY];
        let mut bytes = Vec::new();
        send_activation_packet(&mut bytes, 17, 23, &activations).unwrap();

        let mut decoded = Vec::new();
        let header = recv_activation_packet(&mut Cursor::new(bytes), &mut decoded).unwrap();

        assert_eq!(header.pos, 17);
        assert_eq!(header.seq_len, 23);
        assert_eq!(header.float_count, activations.len() as u32);
        assert_eq!(decoded, activations);
    }

    #[test]
    fn activation_packet_rejects_bad_magic() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());

        let err = recv_activation_packet(&mut Cursor::new(bytes), &mut Vec::new()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("invalid activation header magic"));
    }

    #[test]
    fn activation_packet_rejects_oversized_payload() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&ACTIVATION_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&4_u32.to_le_bytes());

        let err = recv_activation_packet_limited(&mut Cursor::new(bytes), &mut Vec::new(), 3)
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("exceeds limit"));
    }

    #[test]
    fn token_feedback_round_trips() {
        let mut bytes = Vec::new();
        send_token_feedback(&mut bytes, 32001, true).unwrap();

        let feedback = recv_token_feedback(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(feedback.token_id, 32001);
        assert!(feedback.is_finished);
    }

    #[test]
    fn token_feedback_rejects_bad_magic_and_invalid_bool() {
        let mut bad_magic = Vec::new();
        bad_magic.extend_from_slice(&0_u32.to_le_bytes());
        bad_magic.extend_from_slice(&5_u32.to_le_bytes());
        bad_magic.extend_from_slice(&0_u32.to_le_bytes());
        let err = recv_token_feedback(&mut Cursor::new(bad_magic)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);

        let mut invalid_bool = Vec::new();
        invalid_bool.extend_from_slice(&TOKEN_FEEDBACK_MAGIC.to_le_bytes());
        invalid_bool.extend_from_slice(&5_u32.to_le_bytes());
        invalid_bool.extend_from_slice(&2_u32.to_le_bytes());
        let err = recv_token_feedback(&mut Cursor::new(invalid_bool)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}

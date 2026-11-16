//! Encryption for provider API keys stored in the database.
//!
//! Keys set from the dashboard are sealed with XChaCha20-Poly1305 under a key
//! derived from `AEGIS_SECRET_KEY`. The provider name is bound in as
//! associated data, so a row cannot be moved to another provider. Without
//! `AEGIS_SECRET_KEY` nothing can be stored or read, and providers fall back
//! to their environment variables.

use anyhow::{Result, anyhow, bail};
use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use sha2::{Digest, Sha256};

pub const SECRET_ENV: &str = "AEGIS_SECRET_KEY";
const MIN_SECRET_LEN: usize = 32;

pub struct Sealer {
    cipher: XChaCha20Poly1305,
}

pub struct Sealed {
    pub nonce: Vec<u8>,
    pub ciphertext: Vec<u8>,
}

impl Sealer {
    pub fn new(secret: &str) -> Result<Self> {
        let secret = secret.trim();
        if secret.len() < MIN_SECRET_LEN {
            bail!("{SECRET_ENV} must be at least {MIN_SECRET_LEN} characters (try `openssl rand -hex 32`)");
        }
        let key = Sha256::digest(secret.as_bytes());
        Ok(Self { cipher: XChaCha20Poly1305::new(&key) })
    }

    pub fn seal(&self, provider: &str, plaintext: &str) -> Result<Sealed> {
        let mut nonce = [0u8; 24];
        nonce[..16].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
        nonce[16..].copy_from_slice(&uuid::Uuid::new_v4().as_bytes()[..8]);
        let ciphertext = self
            .cipher
            .encrypt(&XNonce::from(nonce), Payload { msg: plaintext.as_bytes(), aad: provider.as_bytes() })
            .map_err(|_| anyhow!("encryption failed"))?;
        Ok(Sealed { nonce: nonce.to_vec(), ciphertext })
    }

    pub fn open(&self, provider: &str, nonce: &[u8], ciphertext: &[u8]) -> Result<String> {
        let Ok(nonce) = <[u8; 24]>::try_from(nonce) else {
            bail!("stored key for {provider} has a malformed nonce");
        };
        let plain = self
            .cipher
            .decrypt(&XNonce::from(nonce), Payload { msg: ciphertext, aad: provider.as_bytes() })
            .map_err(|_| anyhow!("stored key for {provider} cannot be decrypted (was {SECRET_ENV} changed?)"))?;
        Ok(String::from_utf8(plain)?)
    }
}

/// What to show for a stored key: its last four characters, if it is long
/// enough that those reveal nothing useful.
pub fn hint(key: &str) -> String {
    let n = key.chars().count();
    if n < 12 { String::new() } else { key.chars().skip(n - 4).collect() }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &str = "0123456789abcdef0123456789abcdef";

    #[test]
    fn round_trips_and_binds_provider() {
        let s = Sealer::new(SECRET).unwrap();
        let sealed = s.seal("groq", "gsk_example_value").unwrap();
        assert_eq!(s.open("groq", &sealed.nonce, &sealed.ciphertext).unwrap(), "gsk_example_value");
        assert!(s.open("openai", &sealed.nonce, &sealed.ciphertext).is_err());
        let other = Sealer::new("ffffffffffffffffffffffffffffffff").unwrap();
        assert!(other.open("groq", &sealed.nonce, &sealed.ciphertext).is_err());
    }

    #[test]
    fn rejects_short_secret_and_hides_short_keys() {
        assert!(Sealer::new("too-short").is_err());
        assert_eq!(hint("short"), "");
        assert_eq!(hint("gsk_example_value"), "alue");
    }
}

mod auto_encoder;
mod diffusion;
mod super_resolution;
mod text_encoder;

pub use auto_encoder::*;
pub use diffusion::*;
pub use super_resolution::*;
pub use text_encoder::*;

mod clip;
mod esrgan;
mod ldm;
mod swinir;

pub trait Model: Send + Sync {
    fn unload_model(&mut self) {}
}

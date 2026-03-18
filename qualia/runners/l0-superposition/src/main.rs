fn main() {
    #[cfg(feature = "metal")]
    qualia_metal::run_layer(0, "l0-superposition");
    #[cfg(feature = "cuda")]
    qualia_cuda::run_layer(0, "l0-superposition");
}

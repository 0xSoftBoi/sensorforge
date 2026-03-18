fn main() {
    #[cfg(feature = "metal")]
    qualia_metal::run_layer(4, "l4-behavior");
    #[cfg(feature = "cuda")]
    qualia_cuda::run_layer(4, "l4-behavior");
}

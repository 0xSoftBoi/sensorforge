fn main() {
    #[cfg(feature = "metal")]
    qualia_metal::run_layer(5, "l5-behavior");
    #[cfg(feature = "cuda")]
    qualia_cuda::run_layer(5, "l5-behavior");
}

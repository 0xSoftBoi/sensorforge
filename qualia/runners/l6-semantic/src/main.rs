fn main() {
    #[cfg(feature = "metal")]
    qualia_metal::run_layer(6, "l6-semantic");
    #[cfg(feature = "cuda")]
    qualia_cuda::run_layer(6, "l6-semantic");
}

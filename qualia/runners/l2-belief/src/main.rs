fn main() {
    #[cfg(feature = "metal")]
    qualia_metal::run_layer(2, "l2-belief");
    #[cfg(feature = "cuda")]
    qualia_cuda::run_layer(2, "l2-belief");
}

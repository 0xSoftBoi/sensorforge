fn main() {
    #[cfg(feature = "metal")]
    qualia_metal::run_layer(1, "l1-belief");
    #[cfg(feature = "cuda")]
    qualia_cuda::run_layer(1, "l1-belief");
}

fn main() {
    #[cfg(feature = "metal")]
    qualia_metal::run_layer(3, "l3-belief");
    #[cfg(feature = "cuda")]
    qualia_cuda::run_layer(3, "l3-belief");
}

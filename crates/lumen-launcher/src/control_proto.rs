pub mod lumen {
    pub mod control {
        pub mod v1 {
            include!(concat!(env!("OUT_DIR"), "/lumen.control.v1.rs"));
        }
    }
}

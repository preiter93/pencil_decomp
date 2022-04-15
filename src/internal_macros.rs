//! Macros for internal usage
#![allow(unused_macros)]

macro_rules! assert_eq_len {
    ($a: expr, $b: expr, $name: expr) => {
        assert!(
            $a.len() == $b.len(),
            "{:?}: shape mismatch: {:?} /= {:?}",
            $name,
            ($a.len(),),
            ($b.len(),)
        );
    };
}

macro_rules! assert_eq_shape {
    ($a: expr, $b: expr, $name: expr) => {
        assert!(
            $a.shape() == $b.shape(),
            "{:?}: shape mismatch: {:?} /= {:?}",
            $name,
            ($a.shape(),),
            ($b.shape(),)
        );
    };
}

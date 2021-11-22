use crate::nelder_mead::*;

use nalgebra::DVector;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyFunction;


#[pyfunction]
fn nelder_mead(
    obj_fn: &PyFunction,
    x_start: Vec<f64>,
    step: f64,
    no_improve_thr: f64,
    no_improv_break: u64,
    max_iter: u64,
    alpha: f64,
    gamma: f64,
    rho: f64,
    sigma: f64
) -> (Vec<f64>, f64) {

    let x0 = DVector::<f64>::from(x_start);
    let obj_fn_wrp = |x: &DVector<f64>| -> f64 {
        let v: Vec<f64> = x.iter()
            .cloned()
            .collect();
        return obj_fn.call1((v,))
                .unwrap()
                .extract::<f64>()
                .unwrap()
    };

    let results = nelder_mead_algorithm(
        &obj_fn_wrp, 
        x0,
        step, 
        no_improve_thr, 
        no_improv_break, 
        max_iter, 
        alpha, 
        gamma, 
        rho, 
        sigma
    );

    return (
        results.0
            .iter()
            .cloned()
            .collect(), 
        results.1
    )
}

#[pymodule]
fn nelder_mead_optimizer(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nelder_mead, m)?).unwrap();

    Ok(())
}
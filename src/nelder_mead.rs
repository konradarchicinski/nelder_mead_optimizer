use nalgebra::DVector;

/// Finds a local minimum of provided objective function and returns 
/// a tuple containing best parameter vector and best score.
/// 
/// It's a pure Rust implementation of the Nelder-Mead algorithm.
/// Reference: <https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method>
///
/// # Arguments
///
/// * `obj_fn` - function to optimize, must return a scalar score and operate over 
///     a numpy array of the same dimensions as x_start
/// * `x_start` - initial position
/// * `step` - look-around radius in initial step
/// * `no_improve_thr` - threshold informing on no improvement
/// * `no_improv_break` - break after no_improv_break iterations with an
///     improvement lower than no_improv_thr 
/// * `max_iter` - always break after this number of iterations
/// * `alpha` - reflection step parameter, usually equals 1.0
/// * `gamma` - expansion step parameter, usually equals 2.0
/// * `rho` - contraction step parameter, usually equals 0.5
/// * `sigma` - shrink step parameter, usually equals 0.5
///
/// # Examples
///
/// ```
/// use nalgebra::{DVector, dvector};
/// use nelder_mead::nelder_mead;
/// 
/// fn f(x: &DVector<f64>) -> f64 {
///     return x[0].sin() * x[1].cos() * (1.0 / (x[2].abs() + 1.0))
/// }
/// let results = nelder_mead(
///     &f, 
///     dvector![0.0, 0.0, 0.0],
///     0.1,
///     10e-6,
///     10,
///     100,
///     1.0,
///     2.0,
///     -0.5,
///     0.5
/// );
/// 
/// println!("{:?}", results);
/// 
/// assert_eq!(-0.9999447346002792, results.1);
/// ```
/// 
pub fn nelder_mead_algorithm(
    obj_fn: &dyn Fn(&DVector<f64>) -> f64,
    x_start: DVector<f64>,
    step: f64,
    no_improve_thr: f64,
    no_improv_break: u64,
    max_iter: u64,
    alpha: f64,
    gamma: f64,
    rho: f64,
    sigma: f64
) -> (DVector<f64>, f64) {

    // init
    let dim = x_start.len();
    let mut prev_best = obj_fn(&x_start);
    let mut no_improv = 0;
    let mut res = vec![(x_start, prev_best)];

    for i in 0..dim {
        let mut x = res[0].0.clone();
        x[i] += step;
        let score = obj_fn(&x);
        res.push((x, score));
    }

    // simplex iter
    let mut iters = 0;
    loop
    {
        // order
        res.sort_by(|a, b| (a.1).partial_cmp(&b.1).unwrap());
        let best = res[0].1.clone();

        // break after max_iter
        if iters >= max_iter {
            return res[0].clone()
        } 
        iters += 1;

        // break after no_improv_break iterations with no improvement
        println!("Iter {}, best so far: {}", iters, best);

        if best < prev_best - no_improve_thr {
            no_improv = 0;
            prev_best = best;
        } else {
            no_improv += 1;
        }

        if no_improv >= no_improv_break {
            return res[0].clone()
        }

        let last_idx = res.len()-1;

        // centroid
        let mut x0 = DVector::<f64>::zeros(dim);
        for tup in res[..last_idx].iter() {
            for (i, c) in (tup.0).iter().enumerate() {
                x0[i] += c / last_idx as f64;
            }
        }

        // reflection
        let xr = &x0 + alpha*(&x0 - &(res[last_idx].0));
        let rscore = obj_fn(&xr);
        if (res[0].1 <= rscore) & (rscore < res[last_idx-1].1) {
            res.remove(last_idx);
            res.push((xr, rscore));
            continue;
        }

        // expansion
        if rscore < res[0].1 {
            let xe = &x0 + gamma*(&x0 - &(res[last_idx].0));
            let escore = obj_fn(&xe);
            if escore < rscore {
                res.remove(last_idx);
                res.push((xe, escore));
                continue;
            } else {
                res.remove(last_idx);
                res.push((xr, rscore));
                continue;
            }
        }

        // contraction
        let xc = &x0 + rho*(&x0 - &(res[last_idx].0));
        let cscore = obj_fn(&xc);
        if cscore < res[last_idx].1 {
            res.remove(last_idx);
            res.push((xc, cscore));
            continue;
        }

        // reduction
        let x1 = res[0].0.clone();
        let mut nres: Vec<(DVector<f64>, f64)> = vec![];
        for tup in res.iter() {
            let redx = &x1 + sigma*(&tup.0 - &x1);
            let score = obj_fn(&redx);
            nres.push((redx, score)); 
        }
            
        res = nres
    }
}
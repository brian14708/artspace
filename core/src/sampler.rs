use std::{
    collections::{HashMap, VecDeque},
    iter,
};

use crate::model::{Diffusion, DiffusionScheduleParam};

pub struct DdimSampler<'a> {
    pub model: &'a mut dyn Diffusion,
    pub steps: &'a Vec<DiffusionScheduleParam>,
    pub c: HashMap<String, ndarray::ArrayD<f32>>,
    pub seed: ndarray::ArrayD<f32>,
}

impl<'a> DdimSampler<'a> {
    pub fn new(
        model: &'a mut dyn Diffusion,
        steps: &'a Vec<DiffusionScheduleParam>,
        condition: HashMap<String, ndarray::ArrayD<f32>>,
        uncondition: HashMap<String, ndarray::ArrayD<f32>>,
        seed: ndarray::ArrayD<f32>,
    ) -> Self {
        let batch = seed.shape()[0];
        let c: HashMap<_, _> = uncondition
            .into_iter()
            .map(|(k, uncond)| {
                let s = [condition[&k].view()]
                    .into_iter()
                    .chain(iter::repeat(uncond.view()).take(batch))
                    .collect::<Vec<_>>();
                (
                    k,
                    ndarray::concatenate(ndarray::Axis(0), s.as_slice()).unwrap(),
                )
            })
            .collect();

        Self {
            model,
            steps,
            c,
            seed,
        }
    }

    pub fn next(&mut self, i: usize) {
        let t = &self.steps[i];
        let batch = self.seed.shape()[0];

        let seed =
            ndarray::concatenate(ndarray::Axis(0), &[self.seed.view(), self.seed.view()]).unwrap();
        let seed = self.model.execute(&seed, t, &self.c).unwrap();
        let mut e_t = ndarray::ArrayD::<f32>::zeros(self.seed.shape());
        for i in 0..batch {
            e_t.index_axis_mut(ndarray::Axis(0), i).assign(
                &(seed.index_axis(ndarray::Axis(0), i + batch).to_owned()
                    + (seed.index_axis(ndarray::Axis(0), i).to_owned()
                        - seed.index_axis(ndarray::Axis(0), i + batch))
                        * 10.),
            );
        }
        let pred_x0 = (&self.seed - &e_t * ((1. - t.alpha_cumprod).sqrt() as f32))
            / (t.alpha_cumprod.sqrt() as f32);
        let dir_xt = &e_t * ((1. - t.alpha_cumprod_prev).sqrt() as f32);
        // TODO noise
        self.seed = (t.alpha_cumprod_prev.sqrt() as f32) * pred_x0 + dir_xt;
    }
}

// https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lms_discrete.py
pub struct LmsSampler<'a> {
    pub model: &'a mut dyn Diffusion,
    pub steps: &'a Vec<DiffusionScheduleParam>,
    pub c: HashMap<String, ndarray::ArrayD<f32>>,
    pub seed: ndarray::ArrayD<f32>,
    pub derivatives: VecDeque<ndarray::ArrayD<f32>>,
}

impl<'a> LmsSampler<'a> {
    pub fn new(
        model: &'a mut dyn Diffusion,
        steps: &'a Vec<DiffusionScheduleParam>,
        condition: HashMap<String, ndarray::ArrayD<f32>>,
        uncondition: HashMap<String, ndarray::ArrayD<f32>>,
        seed: ndarray::ArrayD<f32>,
    ) -> Self {
        let batch = seed.shape()[0];
        let c: HashMap<_, _> = uncondition
            .into_iter()
            .map(|(k, uncond)| {
                let s = [condition[&k].view()]
                    .into_iter()
                    .chain(iter::repeat(uncond.view()).take(batch))
                    .collect::<Vec<_>>();
                (
                    k,
                    ndarray::concatenate(ndarray::Axis(0), s.as_slice()).unwrap(),
                )
            })
            .collect();

        Self {
            model,
            steps,
            c,
            seed,
            derivatives: VecDeque::new(),
        }
    }

    fn exec(&mut self, t: &DiffusionScheduleParam, sigma: f32) -> ndarray::ArrayD<f32> {
        let mut seed =
            ndarray::concatenate(ndarray::Axis(0), &[self.seed.view(), self.seed.view()]).unwrap();
        seed /= (sigma.powi(2) + 1.).sqrt();
        let batch = self.seed.shape()[0];
        let seed = self.model.execute(&seed, t, &self.c).unwrap();
        let mut e_t = ndarray::ArrayD::<f32>::zeros(self.seed.shape());
        for i in 0..batch {
            e_t.index_axis_mut(ndarray::Axis(0), i).assign(
                &(seed.index_axis(ndarray::Axis(0), i + batch).to_owned()
                    + (seed.index_axis(ndarray::Axis(0), i).to_owned()
                        - seed.index_axis(ndarray::Axis(0), i + batch))
                        * 7.),
            );
        }
        e_t
    }

    fn sigma(t: &DiffusionScheduleParam) -> f64 {
        ((1. - t.alpha_cumprod) / t.alpha_cumprod).sqrt()
    }

    pub fn next(&mut self, i: usize) {
        let t = &self.steps[i];
        let sigma = Self::sigma(t);

        if i == 0 {
            self.seed *= sigma as f32;
        }

        let e_t = self.exec(t, sigma as f32);

        let pred_original_sample = self.seed.to_owned() - sigma as f32 * &e_t;
        let derivative = (self.seed.to_owned() - pred_original_sample) / sigma as f32;
        self.derivatives.push_front(derivative);
        const ORDER: usize = 4;
        if self.derivatives.len() > ORDER {
            self.derivatives.pop_back();
        }

        let order = (i + 1).min(ORDER);
        for (i, coeff) in (0..order)
            .map(|o| self.get_lms_coefficient(order, i, o))
            .enumerate()
            .collect::<Vec<_>>()
        {
            self.seed = &self.seed + &self.derivatives[i] * (coeff as f32);
        }
    }

    fn get_lms_coefficient(&self, order: usize, i: usize, current_order: usize) -> f64 {
        let quad = gauss_quad::GaussLegendre::init(10);

        let a = Self::sigma(&self.steps[i]);
        let b = if i + 1 >= self.steps.len() {
            0.0
        } else {
            Self::sigma(&self.steps[i + 1])
        };
        quad.integrate(a, b, |tau: f64| -> f64 {
            let mut prod = 1.0;
            for k in 0..order {
                if current_order == k {
                    continue;
                }
                prod *= (tau - Self::sigma(&self.steps[i - k]))
                    / (Self::sigma(&self.steps[i - current_order])
                        - Self::sigma(&self.steps[i - k]))
            }
            prod
        })
    }
}

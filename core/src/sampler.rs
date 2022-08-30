use std::{
    collections::{HashMap, VecDeque},
    iter,
};

use quad_rs::prelude::*;

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
        let batch = self.seed.shape()[0];
        let mut e_t = ndarray::ArrayD::<f32>::zeros(self.seed.shape());

        if true {
            for i in 0..batch {
                let s = self
                    .seed
                    .index_axis(ndarray::Axis(0), i)
                    .insert_axis(ndarray::Axis(0))
                    .to_owned()
                    / (sigma.powi(2) + 1.).sqrt();
                let mut c = HashMap::new();
                for (k, v) in self.c.iter() {
                    c.insert(
                        k.to_owned(),
                        v.index_axis(ndarray::Axis(0), i)
                            .insert_axis(ndarray::Axis(0))
                            .to_owned(),
                    );
                }
                let vi = self.model.execute(&s, t, &c).unwrap();

                for (k, v) in self.c.iter() {
                    c.insert(
                        k.to_owned(),
                        v.index_axis(ndarray::Axis(0), i + batch)
                            .insert_axis(ndarray::Axis(0))
                            .to_owned(),
                    );
                }
                let vib = self.model.execute(&s, t, &c).unwrap();

                e_t.index_axis_mut(ndarray::Axis(0), i)
                    .assign(&(vib.to_owned() + (vi - vib) * 7.5).index_axis(ndarray::Axis(0), 0));
            }
            e_t
        } else {
            let mut seed =
                ndarray::concatenate(ndarray::Axis(0), &[self.seed.view(), self.seed.view()])
                    .unwrap();
            seed /= (sigma.powi(2) + 1.).sqrt();
            let seed = self.model.execute(&seed, t, &self.c).unwrap();
            for i in 0..batch {
                e_t.index_axis_mut(ndarray::Axis(0), i).assign(
                    &(seed.index_axis(ndarray::Axis(0), i + batch).to_owned()
                        + (seed.index_axis(ndarray::Axis(0), i).to_owned()
                            - seed.index_axis(ndarray::Axis(0), i + batch))
                            * 7.5),
                );
            }
            e_t
        }
    }

    fn sigma(t: &DiffusionScheduleParam) -> f32 {
        ((1. - t.alpha_cumprod) / t.alpha_cumprod).sqrt() as f32
    }

    pub fn next(&mut self, i: usize) {
        let t = &self.steps[i];
        let sigma = Self::sigma(t);

        if i == 0 {
            self.seed *= sigma;
        }

        let e_t = self.exec(t, sigma);

        let pred_original_sample = self.seed.to_owned() - sigma * &e_t;
        let derivative = (self.seed.to_owned() - pred_original_sample) / sigma;
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
            self.seed = &self.seed + &self.derivatives[i] * coeff;
        }
    }

    fn get_lms_coefficient(&self, order: usize, i: usize, current_order: usize) -> f32 {
        let a = Self::sigma(&self.steps[i]);
        let b = if i + 1 >= self.steps.len() {
            0.0
        } else {
            Self::sigma(&self.steps[i + 1])
        };
        quad_rs::GaussKronrod::default()
            .with_relative_tolerance(1e-4)
            .integrate(
                |tau: f32| -> f32 {
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
                },
                a..b,
                None,
            )
            .unwrap()
            .result
            .unwrap()
    }
}

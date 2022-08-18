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

pub struct PlmsSampler<'a> {
    pub model: &'a mut dyn Diffusion,
    pub steps: &'a Vec<DiffusionScheduleParam>,
    pub c: HashMap<String, ndarray::ArrayD<f32>>,
    pub seed: ndarray::ArrayD<f32>,
    pub eps: VecDeque<ndarray::ArrayD<f32>>,
}

impl<'a> PlmsSampler<'a> {
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
            eps: VecDeque::new(),
        }
    }

    fn exec(
        &mut self,
        x: &ndarray::ArrayD<f32>,
        t: &DiffusionScheduleParam,
    ) -> ndarray::ArrayD<f32> {
        let seed = ndarray::concatenate(ndarray::Axis(0), &[x.view(), x.view()]).unwrap();

        let batch = self.seed.shape()[0];
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
        e_t
    }

    pub fn next(&mut self, i: usize) {
        let t = &self.steps[i];
        let e_t = self.exec(&self.seed.to_owned(), t);

        if self.eps.len() >= 4 {
            self.eps.pop_back();
        }
        self.eps.push_front(e_t);

        let e_t_prime = match self.eps.len() {
            1 => self.eps[0].to_owned(),
            2 => 3. / 2. * self.eps[0].to_owned() - 1. / 2. * &self.eps[1],
            3 => {
                23. / 12. * self.eps[0].to_owned() - 16. / 12. * &self.eps[1]
                    + 5. / 12. * &self.eps[2]
            }
            4 => {
                55. / 24. * self.eps[0].to_owned() - 59. / 24. * &self.eps[1]
                    + 37. / 24. * &self.eps[2]
                    - 9. / 24. * &self.eps[3]
            }
            _ => unreachable!(),
        };

        let prev_over_alpha_sqrt = (t.alpha_cumprod_prev / t.alpha_cumprod).sqrt();
        self.seed = prev_over_alpha_sqrt as f32 * &self.seed
            + ((1. - t.alpha_cumprod_prev).sqrt()
                - prev_over_alpha_sqrt * (1. - t.alpha_cumprod).sqrt()) as f32
                * &e_t_prime;
    }
}

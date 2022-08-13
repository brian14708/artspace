use std::collections::HashMap;

use ndarray::IxDyn;

use crate::model::{Diffusion, DiffusionScheduleParam};

pub struct DdimSampler<'a> {
    pub model: &'a mut dyn Diffusion,
    pub c: HashMap<String, ndarray::ArrayD<f32>>,
    pub seed: ndarray::ArrayD<f32>,
}

impl<'a> DdimSampler<'a> {
    pub fn new(
        model: &'a mut dyn Diffusion,
        condition: HashMap<String, ndarray::ArrayD<f32>>,
        uncondition: HashMap<String, ndarray::ArrayD<f32>>,
        seed: ndarray::ArrayD<f32>,
    ) -> Self {
        let mut c = HashMap::new();
        for (k, v) in uncondition {
            let tmp = condition[&k].view();
            let vv = v.view();
            let s = (0..seed.shape()[0])
                .map(|i| tmp.index_axis(ndarray::Axis(0), i))
                .into_iter()
                .chain((0..seed.shape()[0]).map(|_| vv.index_axis(ndarray::Axis(0), 0)))
                .collect::<Vec<_>>();
            let s = ndarray::stack(ndarray::Axis(0), s.as_slice()).unwrap();
            c.insert(k, s.to_owned());
        }

        DdimSampler { model, c, seed }
    }

    pub fn next(&mut self, t: &DiffusionScheduleParam) {
        let seed = ndarray::stack(ndarray::Axis(0), &[self.seed.view(), self.seed.view()]).unwrap();
        let shape: Vec<_> = [seed.shape()[0] * seed.shape()[1]]
            .iter()
            .chain(&seed.shape()[2..])
            .cloned()
            .collect();
        let seed = seed.into_shape(IxDyn(shape.as_slice())).unwrap();

        let seed = self.model.execute(&seed, t, &self.c).unwrap();
        let batch = self.seed.shape()[0];
        let mut e_t = ndarray::ArrayD::<f32>::zeros(self.seed.shape());
        for i in 0..batch {
            e_t.index_axis_mut(ndarray::Axis(0), i).assign(
                &(seed.index_axis(ndarray::Axis(0), i + batch).to_owned()
                    + (seed.index_axis(ndarray::Axis(0), i).to_owned()
                        - seed.index_axis(ndarray::Axis(0), i + batch))
                        * 5.),
            );
        }
        let pred_x0 = (&self.seed - &e_t * ((1. - t.alpha_cumprod).sqrt() as f32))
            / (t.alpha_cumprod.sqrt() as f32);
        let dir_xt = &e_t * ((1. - t.alpha_cumprod_prev).sqrt() as f32);
        // TODO noise
        self.seed = (t.alpha_cumprod_prev.sqrt() as f32) * pred_x0 + dir_xt;
    }
}

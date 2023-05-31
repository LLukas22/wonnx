use wonnx::builder::*;
use std::collections::HashMap;

async fn run(){
    let a = tensor("x", &[1, 3], vec![0.1, 0.2, 0.3].into());
    let b = tensor("y", &[1, 3], vec![3.0, 2.0, 1.0].into());
    let axb = a.add(&b);
    let neg_a = a.neg();

    let sesh = session_for_outputs(&["a+b","neg_a"], &[axb,neg_a], 13).await.unwrap();
    let result = sesh.run(&HashMap::new()).await.unwrap();
    println!("result: {:?}", result);
}

fn main() {
    pollster::block_on(run());
}
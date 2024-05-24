use tch::nn::ModuleT;
use tch::vision::{
    imagenet, resnet,
};
use axum::{
    response::{Html},
    routing::{get, post},
    extract::multipart::{Multipart},
    Router,
};
use std::sync::OnceLock;
use std::env::var;
use serde::Serialize;
use minijinja::render;

const DEFAULT_MODEL_PATH: &'static str = "/data";

const SUBMISSION_TEMPLATE: &'static str = r#"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Is it a fox?</title>
</head>

<body>
    <h1>🦊 Is it a fox?</h1>
    <form method="post" enctype="multipart/form-data" action="/check">
        <p>
           <label>Send an image file: </label><br/>
           <input type="file" name="file"/>
        </p>
        <p>
            <input type="submit" value="Let's find out"/>
        </p>
    </form>
</body>
</html>
"#;

const IDENTIFIED_TEMPLATE: &'static str = r#"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Was it a fox?</title>
</head>

<body>
    {% if result.success %}
    <h1>🎉🦊 It was a fox! 🦊🎉</h1>
    {% else %}
    <h1>😥 It wasn't a fox..</h1>
    {% endif %}
    {% if result.object is not none %}
    <p>We thought it was a <b>{{ result.object.name }}</b> with a
    probability of <b>{{ result.object.probability|round }}%</b></p>
    {% else %}
    <p>We couldn't even figure out what it was..</p>
    {% endif %}

    <a href="/">Go Back</a>
</body>
</html>
"#;

/// Represents the result of an identifier attempt
#[derive(Debug,Serialize)]
struct WasItAFoxResult {
    success: bool,
    object: Option<IdentifiedObject>,
}
impl Default for WasItAFoxResult {
    fn default() -> Self {
        WasItAFoxResult{
            success: false,
            object: None,
        }
    }
}

/// Represents an identified object from a supplied image
#[derive(Debug,Serialize)]
struct IdentifiedObject {
    /// Probability represents the probability percentage that the image
    /// is the identified object
    probability: f64,
    /// Name represents the textual representation of the identified object
    name: String,
}

/// Handler for the default form input view
async fn file_submission() -> Html<String> {
    let r = render!(SUBMISSION_TEMPLATE);
    Html(r)
}

/// Handler for the object identifier view
async fn is_it_a_fox(mut multipart: Multipart) -> Html<String> {
    let mut results:Vec<(String, f64)> = Vec::new();
    let mut found_a_fox: bool = false;

    let model_path = var("MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
    while let Some(mut field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap().to_string();
        let data = field.bytes().await.unwrap();
        let image = imagenet::load_image_and_resize224_from_memory(data.as_ref()).unwrap();
        let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let net: Box<dyn ModuleT> = Box::new(resnet::resnet34(&vs.root(), imagenet::CLASS_COUNT));
        vs.load(format!("{}/resnet34.ot",model_path)).unwrap();
        let output =
            net.forward_t(&image.unsqueeze(0), /* train= */ false).softmax(-1, tch::Kind::Float); // Convert to probability.
        for (probability, class) in imagenet::top(&output, 5).iter() {
            if class.contains("fox") || class.contains("vulpes") {
                found_a_fox = true;
            }
            results.push((class.to_string(), 100.0*probability));
        }
    }

    let mut result = WasItAFoxResult::default();
    result.success = found_a_fox;
    if results.len() > 0 {
        let (name,probability) = results.get(0).unwrap();
        result.object = Some(IdentifiedObject{
            name: name.to_string(),
            probability: probability.clone(),
        });
    }
    let r = render!(IDENTIFIED_TEMPLATE, result => result);
    Html(r)
}

#[tokio::main]
async fn main() {
    // Define Routes
    let app = Router::new()
        .route("/", get(file_submission))
        .route("/check", post(is_it_a_fox));

    println!("Running on http://localhost:3000");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
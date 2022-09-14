use gloo_net::http::Request;
use serde::Deserialize;
use wasm_bindgen::JsValue;
use web_sys::{Blob, File, FileList, FormData, HtmlInputElement};
use yew::prelude::*;

#[derive(Clone, PartialEq, Deserialize)]
struct ApiResponse {
    prediction: String,
}

#[function_component(FileUploadForm)]
fn file_upload_form() -> Html {
    let input_ref = use_node_ref();
    let prediction = use_state(|| "".to_string());

    let on_form_submit: Callback<FocusEvent> = {
        let input_ref = input_ref.clone();

        let prediction = prediction.clone();

        Callback::from(move |e: FocusEvent| {
            e.prevent_default();

            let prediction = prediction.clone();

            if let Some(input) = input_ref.cast::<HtmlInputElement>() {
                let files: FileList = input.files().unwrap();
                let file: File = files.get(0).unwrap();
                let form_data: Result<FormData, JsValue> = FormData::new();

                if let Ok(f_data) = form_data {
                    let blob: Result<Blob, JsValue> = file.slice();

                    if let Ok(b) = blob {
                        let filled_form: Result<(), JsValue> = f_data.append_with_blob("img", &b);

                        if let Ok(_filled_form) = filled_form {
                            wasm_bindgen_futures::spawn_local(async move {
                                let res: ApiResponse =
                                    Request::post("http://127.0.0.1:5000/predict")
                                        .body(f_data)
                                        .send()
                                        .await
                                        .unwrap()
                                        .json()
                                        .await
                                        .unwrap();

                                log::info!("{}", res.prediction);
                                prediction.set(res.prediction);
                            });
                        }
                    }
                };
            }
        })
    };

    html! {
        <>
            <form onsubmit={on_form_submit}>
                <h4>{"Add image"}</h4>
                <input type="file" ref={&input_ref}/>
                <input type="submit" />
            </form>
            <div>{ (*prediction).clone()}</div>
        </>
    }
}

#[function_component(App)]
fn app() -> Html {
    html! {
        <>
            <h1>{ "Parkinsons Classifier" }</h1>
            <FileUploadForm/>
        </>
    }
}

fn main() {
    wasm_logger::init(wasm_logger::Config::default());
    yew::start_app::<App>();
}

use std::panic;

use gloo_net::http::Request;
use wasm_bindgen::JsValue;
use web_sys::{Blob, File, FileList, FormData, HtmlInputElement};
use yew::prelude::*;

#[function_component(FileUploadForm)]
fn file_upload_form() -> Html {
    let input_ref = use_node_ref();

    let on_form_submit: Callback<FocusEvent> = {
        let input_ref = input_ref.clone();

        Callback::from(move |e: FocusEvent| {
            e.prevent_default();

            if let Some(input) = input_ref.cast::<HtmlInputElement>() {
                let files: Option<FileList> = input.files();
                match files {
                    Some(f_list) => select_file(f_list),
                    None => panic!("Error loading file list"),
                }
            }
        })
    };

    html! {
        <form onsubmit={on_form_submit}>
            <h4>{"Add image"}</h4>
            <input type="file" ref={&input_ref}/>
            <input type="submit" />
        </form>
    }
}

fn select_file(f_list: FileList) {
    let file = f_list.get(0);

    match file {
        Some(f) => create_form_data(f),
        None => panic!("Error getting file"),
    }
}

fn create_form_data(f: File) {
    let form_data: Result<FormData, JsValue> = FormData::new();

    if let Ok(f_data) = form_data {
        fill_form(f_data, f)
    };
}

fn fill_form(f_data: FormData, f: File) {
    let blob: Result<Blob, JsValue> = f.slice();

    if let Ok(b) = blob {
        let filled_form: Result<(), JsValue> = f_data.append_with_blob("img", &b);

        if let Ok(_filled_form) = filled_form {
            log::info!("Form filled, {:?}", f_data);

            wasm_bindgen_futures::spawn_local(async move {
                let res = Request::post("http://127.0.0.1:5000/predict")
                    .body(f_data)
                    .send()
                    .await
                    .unwrap()
                    .text()
                    .await
                    .unwrap();

                log::info!("{:?}", res);
            });
        }
    }
}

#[function_component(App)]
fn app() -> Html {
    html! {
        <>
            <h1>{ "Parkinsons Classifier" }</h1>
            <FileUploadForm />
        </>
    }
}

fn main() {
    wasm_logger::init(wasm_logger::Config::default());
    yew::start_app::<App>();
}

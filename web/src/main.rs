use gloo_net::http::Request;
use serde::Deserialize;
use wasm_bindgen::JsValue;
use web_sys::{Blob, File, FileList, FormData, HtmlInputElement, Url};
use yew::prelude::*;

#[derive(Clone, PartialEq, Deserialize)]
struct ApiResponse {
    prediction: String,
}

#[function_component(FileUploadForm)]
fn file_upload_form() -> Html {
    let input_ref = use_node_ref();
    let prediction = use_state(|| "".to_string());
    let file_name = use_state(|| "".to_string());
    let image_preview = use_state(|| "".to_string());

    let on_file_add: Callback<Event> = {
        let input_ref = input_ref.clone();
        let file_name = file_name.clone();
        let image_preview = image_preview.clone();
        let prediction = prediction.clone();

        Callback::from(move |_e: Event| {
            let input_ref = input_ref.clone();
            let file_name = file_name.clone();
            let image_preview = image_preview.clone();
            let prediction = prediction.clone();

            if let Some(input) = input_ref.cast::<HtmlInputElement>() {
                let file = get_file_obj(&input).unwrap();
                let b = get_img_blob(&input).unwrap();

                let d_url = Url::create_object_url_with_blob(&b).unwrap();

                image_preview.set(d_url);
                prediction.set("".to_string());
                file_name.set(file.name());
            }
        })
    };

    let on_form_submit: Callback<FocusEvent> = {
        let input_ref = input_ref.clone();
        let prediction = prediction.clone();

        Callback::from(move |e: FocusEvent| {
            e.prevent_default();
            let prediction = prediction.clone();

            if let Some(input) = input_ref.cast::<HtmlInputElement>() {
                let blob = get_img_blob(&input);
                let blob = match blob {
                    Ok(b) => b,
                    Err(_) => return,
                };

                let form_data_res: Result<FormData, JsValue> = FormData::new();

                let f_data = match form_data_res {
                    Ok(fd) => fd,
                    Err(_) => return,
                };

                let filled_form_res: Result<(), JsValue> = f_data.append_with_blob("img", &blob);

                match filled_form_res {
                    Ok(_) => {
                        wasm_bindgen_futures::spawn_local(async move {
                            let res = do_request(f_data).await;
                            log::info!("prediction: {}", res.prediction);
                            prediction.set(res.prediction);
                        });
                    }
                    Err(_) => return,
                }
            }
        })
    };

    fn get_file_obj(input: &HtmlInputElement) -> Result<File, ()> {
        let files: FileList = input.files().unwrap();
        let file: File = files.get(0).unwrap();

        Ok(file)
    }

    fn get_img_blob(input: &HtmlInputElement) -> Result<Blob, ()> {
        let file = get_file_obj(&input).unwrap();
        let blob_res: Result<Blob, JsValue> = file.slice();

        match blob_res {
            Ok(ff) => Ok(ff),
            Err(_) => Err(()),
        }
    }

    async fn do_request(f_data: FormData) -> ApiResponse {
        Request::post("http://127.0.0.1:5000/predict")
            .body(f_data)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    html! {
        <>
            <form onsubmit={on_form_submit}>
                <h4 class="subtitle">{"Add image"}</h4>

                <div class="m-2">
                    <div class="file">
                        <label class="file-label">
                            <input class="file-input" type="file" ref={&input_ref} onchange={on_file_add}/>
                            <span class="file-cta">
                            <span class="file-icon">
                                <i class="fas fa-upload"></i>
                            </span>
                            <span class="file-label">
                                { "Choose a file" }
                            </span>
                            </span>
                            <span class="file-name">
                            { (*file_name).clone() }
                            </span>
                        </label>
                    </div>
                </div>

                <div class="m-2">
                    <img src={ (*image_preview).clone() } />
                </div>

                <input type="submit" value="predict" class="button is-primary" />
            </form>
            <div class="m-2">
                <p class="is-size-4">
                    { (*prediction).clone() }
                </p>
            </div>
        </>
    }
}

#[function_component(App)]
fn app() -> Html {
    html! {
        <div class="is-flex is-justify-content-center">
            <div class="is-flex is-flex-direction-column">
                <h1 class="title">{ "Parkinsons Classifier" }</h1>
                <FileUploadForm/>
            </div>
        </div>
    }
}

fn main() {
    wasm_logger::init(wasm_logger::Config::default());
    yew::start_app::<App>();
}

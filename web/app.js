//the page talks to the FastAPI server with fetch().
const fileInput = document.querySelector("#file-input");
//the text that tells the user to choose a photo
const pickerText = document.querySelector("#picker-text");
const preview = document.querySelector("#preview");
const form = document.querySelector("#upload-form");
const submitButton = document.querySelector("#submit-button");
const result = document.querySelector("#result");
const resultLabel = document.querySelector("#result-label");
const resultBar = document.querySelector("#result-bar");
const resultConfidence = document.querySelector("#result-confidence");
const errorBox = document.querySelector("#error");
const statsBox = document.querySelector("#stats");
const historyBody = document.querySelector("#history");
//the id of the last photo sent, so its result can be shown when a worker finishes it
let lastId = null;
//when a photo is chosen: show it, before anything is sent to the server.
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    lastId = null;
    pickerText.textContent = file.name;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    submitButton.disabled = false;
    result.hidden = true;
    errorBox.hidden = true;
});
//when "Classify" is pressed: send the photo to POST /predict.
form.addEventListener("submit", async (event) => {
    event.preventDefault(); // stops the browser from reloading the page
    const file = fileInput.files[0];
    if (!file) return;
    //FormData packs the photo the same way an HTML form would.
    const data = new FormData();
    data.append("file", file);
    submitButton.disabled = true;
    submitButton.textContent = "Sending…";
    errorBox.hidden = true;
    try{
        const response = await fetch("/predict", { method: "POST", body: data });
        const body = await response.json().catch(() => ({}));
        if (!response.ok){
            throw new Error(body.detail || `The server answered with error ${response.status}.`);
        }
        lastId = body.id;
        showResult(body);
        await refresh();
    } 
    catch (error){
        errorBox.textContent = error.message;
        errorBox.hidden = false;
    } 
    finally{
        submitButton.disabled = false;
        submitButton.textContent = "Classify";
    }
});
function showResult(prediction) {
    result.hidden = false;
    if (prediction.status === "done"){
        const percent = (prediction.confidence * 100).toFixed(1);
        resultLabel.textContent = prediction.label === "cat" ? "Cat" : "Dog";
        resultBar.style.width = percent + "%";
        resultConfidence.textContent = `${percent}% sure, classified by worker ${prediction.worker}`;
    }
    else if (prediction.status === "processing"){
        resultLabel.textContent = "Classifying…";
        resultBar.style.width = "0";
        resultConfidence.textContent = `Worker ${prediction.worker} is on it.`;
    }
    else if (prediction.status === "failed"){
        resultLabel.textContent = "Failed";
        resultBar.style.width = "0";
        resultConfidence.textContent = "No worker could classify this photo.";
    }
    else{
        resultLabel.textContent = "In the queue";
        resultBar.style.width = "0";
        resultConfidence.textContent = "Waiting for a free worker.";
    }
}
async function getJSON(url) {
    const response = await fetch(url);
    return response.json();
}
async function refresh() {
    showHistory(await getJSON("/predictions?limit=10"));
    showStats(await getJSON("/stats"));
}
function showHistory(predictions) {
    historyBody.replaceChildren();
    for (const p of predictions){
        const row = document.createElement("tr");
        //until a worker is done, the label column shows the status instead: 
        //processing/pending/done/failed
        const done = p.status === "done";
        const cells = [
            new Date(p.created_at).toLocaleString(),
            p.filename,
            done ? p.label : p.status,
            p.worker || "",
            done ? (p.confidence * 100).toFixed(1) + "%" : "",
        ];
        cells.forEach((text, i) => {
            const cell = document.createElement("td");
            cell.textContent = text;
            //i represents the position in cells
            //0: time
            //1: photo
            //2: label
            //3: worker
            //4: confidence
            if (i === 1) cell.className = "file";
            if (i === 3) cell.className = "worker";
            row.append(cell);
        });
        historyBody.append(row);
    }
}
function showStats(stats) {
    if (stats.length === 0) {
        statsBox.textContent = "No predictions yet.";
        return;
    }
    statsBox.textContent = stats
        .map((s) => {
            const name = s.label === "cat" ? "Cats" : "Dogs";
            return `${name}: ${s.count}, average confidence ${(s.avg_confidence * 100).toFixed(1)}%`;
        })
        .join(" · ");
}
//every second, ask the server again, so the page shows when a worker finishes a photo
async function update() {
    await refresh();
    if (lastId !== null) {
        showResult(await getJSON(`/predictions/${lastId}`));
    }
}
refresh();
setInterval(update, 1000);

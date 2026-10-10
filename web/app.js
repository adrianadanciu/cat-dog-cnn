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
//when a photo is chosen: show it, before anything is sent to the server.
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
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
    submitButton.textContent = "Classifying…";
    errorBox.hidden = true;
    try{
        const response = await fetch("/predict", { method: "POST", body: data });
        const body = await response.json().catch(() => ({}));
        if (!response.ok){
            throw new Error(body.detail || `The server answered with error ${response.status}.`);
        }
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
    const percent = (prediction.confidence * 100).toFixed(1);
    resultLabel.textContent = prediction.label === "cat" ? "Cat" : "Dog";
    resultBar.style.width = percent + "%";
    resultConfidence.textContent = `${percent}% sure`;
    result.hidden = false;
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
        const cells = [
            new Date(p.created_at).toLocaleString(),
            p.filename,
            p.label,
            (p.confidence * 100).toFixed(1) + "%",
        ];
        cells.forEach((text, i) => {
            const cell = document.createElement("td");
            cell.textContent = text;
            if (i === 1) cell.className = "file";
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
refresh();

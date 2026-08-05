const API_BASE = "http://localhost:8000";

const uploadInput = document.getElementById("uploadInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");

const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const searchStatus = document.getElementById("searchStatus");

const resultsDiv = document.getElementById("results");

uploadBtn.addEventListener("click", async () => {
  const files = uploadInput.files;
  if (!files.length) {
    uploadStatus.textContent = "Please select at least one image.";
    return;
  }

  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  uploadStatus.textContent = `Uploading and indexing ${files.length} image(s)...`;

  try {
    const response = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    uploadStatus.textContent =
      `Done. Processed ${data.images_processed} image(s), ` +
      `indexed ${data.faces_indexed} face(s). Total in index: ${data.index_size}.`;
  } catch (err) {
    uploadStatus.textContent = "Upload failed: " + err.message;
  }
});

searchBtn.addEventListener("click", async () => {
  const files = searchInput.files;
  if (!files.length) {
    searchStatus.textContent = "Please select a query face image.";
    return;
  }

  const formData = new FormData();
  formData.append("file", files[0]);

  searchStatus.textContent = "Searching...";
  resultsDiv.innerHTML = "";

  try {
    const response = await fetch(`${API_BASE}/search`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (data.error) {
      searchStatus.textContent = data.error;
      return;
    }

    searchStatus.textContent = `Found ${data.matches.length} matching photo(s).`;

    for (const match of data.matches) {
      const figure = document.createElement("figure");
      const img = document.createElement("img");
      img.src = API_BASE + match.thumbnail_url;
      img.alt = match.filename;

      const caption = document.createElement("figcaption");
      caption.textContent = `${match.filename} (distance: ${match.distance})`;

      figure.appendChild(img);
      figure.appendChild(caption);
      resultsDiv.appendChild(figure);
    }
  } catch (err) {
    searchStatus.textContent = "Search failed: " + err.message;
  }
});

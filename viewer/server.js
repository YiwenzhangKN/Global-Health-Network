// Small web server for the network viewer.
// Run inside this folder:  npm install   then   npm start   ->  http://localhost:3000

const express = require("express");
const path = require("path");

// Config +++
const PORT = 5000;
const NETWORK_FILE = path.join(__dirname, "..", "data", "data", "network", "network.json"); // written by network_builder.py
// Config ---

const app = express();

// network.html asks for "network.json" -> send the file that network_builder.py exported
app.get("/network.json", (req, res) => res.sendFile(NETWORK_FILE));

// the viewer itself
app.get("/", (req, res) => res.sendFile(path.join(__dirname, "network.html")));
app.use(express.static(__dirname));

app.listen(PORT, () => console.log(`Network viewer running at http://localhost:${PORT}`));
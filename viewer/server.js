require('dotenv').config(); //Load .env

const express = require("express");
const path = require("path");

const PORT = process.env.PORT || 5000;
const NETWORK_DIR = path.join(__dirname, "..", "data", "data", "network"); //Data Sourcedir

const app = express();
app.use(express.static(__dirname)); //Static-serve network files

//JSON loader
app.get("/network_:method.json", (req, res) => res.sendFile(path.join(NETWORK_DIR, `network_${req.params.method}.json`)));

// Dashboard Routing
app.get("/", (req, res) => res.sendFile(path.join(__dirname, "network.html")));

app.listen(PORT, () => console.log(`Network viewer running at http://localhost:${PORT}`));
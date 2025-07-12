// src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css'; // You can add global styles here or remove if not needed.
import App from './App';

// Create a root.
const root = ReactDOM.createRoot(document.getElementById('root'));

// Render the app inside StrictMode.
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
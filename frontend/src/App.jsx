import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Monitor from './pages/Monitor';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/monitor" element={<Monitor />} />
    </Routes>
  );
}

export default App;

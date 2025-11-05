import React, { useState } from 'react';
import './App.css'; // Our beautiful CSS

function App() {
  // 1. New State for the new features
  const [formData, setFormData] = useState({
    credit_score: 650,
    age: 40,
    tenure: 5,
    balance: 100000,
    products_number: 1,
    estimated_salary: 150000,
    country: 'France',
    gender: 'Male',
    active_member: 1, // Use 1 for 'Yes'
  });
  
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    
    // Handle number, text, and select inputs
    let val = value;
    if (type === 'number') {
      val = parseFloat(value);
    }
    if (name === 'active_member') {
      val = parseInt(value); // Make sure this is an integer 1 or 0
    }

    setFormData((prev) => ({
      ...prev,
      [name]: val,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setPrediction(null);
    setError(null);

    const API_URL = 'https://cloud-class-api.onrender.com/predict_churn';

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData), // Send formData directly
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || 'Network response was not ok');
      }

      const data = await response.json();
      if (data.error) {
        setError(data.error);
        setPrediction(null);
      } else {
        setPrediction(data);
        setError(null);
      }
    } catch (err) {
      setError(`Prediction failed: ${err.message}`);
      console.error('Fetch error:', err);
    }
  };
  
  const getResultStyle = () => {
    if (!prediction) return {};
    const prob = prediction.churn_probability;
    if (prob > 0.7) return { color: 'red', fontWeight: 'bold' };
    if (prob > 0.4) return { color: 'orange', fontWeight: 'bold' };
    return { color: 'green', fontWeight: 'bold' };
  };

  return (
    <div className="App">
      <main>
        <header className="App-header">
          <h1>Bank Customer Churn Predictor</h1>
        </header>
        
        {/* 2. New Form for the new features */}
        <form onSubmit={handleSubmit}>
          {/* --- Numerical Inputs --- */}
          <div>
            <label>Credit Score:</label>
            <input type="number" name="credit_score" value={formData.credit_score} onChange={handleChange} />
          </div>
          <div>
            <label>Age:</label>
            <input type="number" name="age" value={formData.age} onChange={handleChange} />
          </div>
          <div>
            <label>Tenure (Years):</label>
            <input type="number" name="tenure" value={formData.tenure} onChange={handleChange} />
          </div>
          <div>
            <label>Balance:</label>
            <input type="number" step="0.01" name="balance" value={formData.balance} onChange={handleChange} />
          </div>
          <div>
            <label>Number of Products:</label>
            <input type="number" name="products_number" value={formData.products_number} onChange={handleChange} />
          </div>
          <div>
            <label>Estimated Salary:</label>
            <input type="number" step="0.01" name="estimated_salary" value={formData.estimated_salary} onChange={handleChange} />
          </div>
          
          {/* --- Categorical Inputs --- */}
          <div>
            <label>Country:</label>
            <select name="country" value={formData.country} onChange={handleChange}>
              <option value="India">India</option>
              <option value="Italy">Italy</option>
              <option value="Spain">Spain</option>
              <option value="Germany">Germany</option>
            </select>
          </div>
          <div>
            <label>Gender:</label>
            <select name="gender" value={formData.gender} onChange={handleChange}>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </div>
          <div>
            <label>Active Member:</label>
            <select name="active_member" value={formData.active_member} onChange={handleChange}>
              <option value={1}>Yes</option>
              <option value={0}>No</option>
            </select>
          </div>
          
          {/* Empty div for grid alignment, if needed */}
          <div></div>

          <div className="button-container">
            <button type="submit">Calculate Churn Risk</button>
          </div>
        </form>

        <div className="result">
          {error && <p className="error">{error}</p>}
          {prediction && (
            <div>
              <h2>Prediction Result:</h2>
              <p style={getResultStyle()}>
                Churn Probability: {(prediction.churn_probability * 100).toFixed(2)}%
              </p>
              <p>Likely to Churn: {prediction.prediction_label}</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
import React, { useEffect, useState } from "react";
import Select from "react-select";
import "./App.css";

const customStyles = {
  control: (provided) => ({
    ...provided,
    borderRadius: '8px',
    padding: '2px',
    borderColor: '#d1d5db',
    boxShadow: 'none',
    fontSize: '14px'
  }),
  option: (provided, state) => ({
    ...provided,
    backgroundColor: state.isSelected ? '#2563eb' : 'white',
    color: state.isSelected ? 'white' : '#111827',
    padding: 10,
  }),
};

function App() {
  const [vehicleOptions, setVehicleOptions] = useState([]);
  const [vehicleType, setVehicleType] = useState(null);
  const [locations, setLocations] = useState([]);
  const [startPoint, setStartPoint] = useState(null);
  const [endPoint, setEndPoint] = useState(null);
  const [mapUrl, setMapUrl] = useState("http://localhost:8000/map");
  const [results, setResults] = useState(null);

  useEffect(() => {
    const fetchLocations = async () => {
      const res = await fetch("http://localhost:8000/locations");
      const data = await res.json();
      setLocations(data);
    };
    fetchLocations();
  }, []);

  useEffect(() => {
    const fetchVehicles = async () => {
      const res = await fetch("http://localhost:8000/vehicles");
      const data = await res.json();
      setVehicleOptions(data);
      if (data.length > 0) setVehicleType(data[0]);
    };
    fetchVehicles();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    // Clear previous results before loading new ones
    setResults(null);

    if (!startPoint || !endPoint) {
      alert("Please select both start and end points.");
      return;
    }

    try {
      const query = `vehicle=${vehicleType.value}&start=${startPoint.value}&end=${endPoint.value}`;
      const response = await fetch(`http://localhost:8000/route?${query}`);
      const data = await response.json();

      setResults(data);
      setMapUrl(`http://localhost:8000/map?ts=${Date.now()}`); // refresh map
    } catch (error) {
      console.error("Error fetching route:", error);
    }
  };

  return (
    <div className="container">
      <div className="sidebar">
        <h2>EV Route</h2>
        <form onSubmit={handleSubmit}>
          <label htmlFor="vehicleType">Vehicle Type</label>
          <Select
            options={vehicleOptions}
            value={vehicleType}
            onChange={setVehicleType}
            styles={customStyles}
            placeholder="Select vehicle type"
            isSearchable
          />

          <label>Start Location</label>
          <Select
            options={locations}
            value={startPoint}
            onChange={setStartPoint}
            styles={customStyles}
            placeholder="Enter start location"
            isSearchable
          />

          <label>End Location</label>
          <Select
            options={locations}
            value={endPoint}
            onChange={setEndPoint}
            styles={customStyles}
            placeholder="Enter end location"
            isSearchable
          />

          <button type="submit">Find Route</button>
        </form>

        {results && (
          <div className="results">
            <h3>Results:</h3>
            <p>
              <strong>Total Length (Length Path):</strong> {results.total_length_len} {results.length_type_len}
            </p>
            <p>
              <strong>Total Length (Battery Path):</strong> {results.total_length_bat} {results.length_type_bat}
            </p>
            <p>
              <strong>Battery Usage (Length Path):</strong>{" "}
              {results.total_battery_len.toFixed(2)} Wh
            </p>
            <p>
              <strong>Battery Usage (Battery Path):</strong>{" "}
              {results.total_battery_bat.toFixed(2)} Wh
            </p>
          </div>
        )}
      </div>

      <div className="map-container">
        <iframe
          title="EV Map"
          src={mapUrl}
          className="map-iframe"
          frameBorder="0"
        ></iframe>
      </div>
    </div>
  );
}

export default App;

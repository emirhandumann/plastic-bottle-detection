import axios from 'axios';

// Raspberry Pi'nin IP adresi
const API_URL = 'http://172.20.10.3:5000/api';

export const captureImage = async () => {
  try {
    const response = await axios.get(`${API_URL}/capture`);
    return response.data;
  } catch (error) {
    console.error('Capture error:', error);
    throw error;
  }
};

export const detectBottles = async (imageData: string) => {
  try {
    const response = await axios.post(`${API_URL}/detect`, {
      image: imageData
    });
    return response.data;
  } catch (error) {
    console.error('Detection error:', error);
    throw error;
  }
}; 
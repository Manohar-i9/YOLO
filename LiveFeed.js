import React, { useState, useEffect } from "react";
import { View, Text, StyleSheet, Image, Button, ScrollView } from "react-native";
import { WebView } from 'react-native-webview';

// Configure camera streams with their own IPs
const cameras = [
  { id: 1, title: "WEB CAM - HOME", url: "http://172.20.10.3:8000/live" },
];

export default function LiveFeed() {
  const [imageUris, setImageUris] = useState({});
  const [webViewVisible, setWebViewVisible] = useState({});

  // Fetch image every 100ms for each camera
  useEffect(() => {
    const intervals = cameras.map((cam) => {
      return setInterval(() => {
        setImageUris((prev) => ({
          ...prev,
          [cam.id]: `${cam.url}?t=${new Date().getTime()}`,
        }));
      }, 100);
    });

    return () => intervals.forEach(clearInterval); // Clear intervals on unmount
  }, []);

  const toggleWebView = (camId) => {
    setWebViewVisible((prev) => ({
      ...prev,
      [camId]: !prev[camId],
    }));
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      {cameras.map((cam) => (
        <View key={cam.id} style={styles.cameraContainer}>
          <Text style={styles.title}>{cam.title}</Text>
          <Text style={styles.subtitle}>Streaming...</Text>

          <View style={styles.buttonContainer}>
            <Button
              title={webViewVisible[cam.id] ? "Hide Stream" : "Show Stream"}
              onPress={() => toggleWebView(cam.id)}
            />
          </View>

          {webViewVisible[cam.id] ? (
            <WebView
              source={{ uri: cam.url }}
              style={styles.webview}
              onError={(e) => console.log(`Error loading page (Cam ${cam.id}): `, e.nativeEvent)}
              onLoad={() => console.log(`WebView for Cam ${cam.id} loaded successfully`)}
            />
          ) : (
            <View>
              {imageUris[cam.id] ? (
                <Image
                  source={{ uri: imageUris[cam.id] }}
                  style={styles.image}
                />
              ) : (
                <Text>Loading...</Text>
              )}
            </View>
          )}
        </View>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingVertical: 50,
    backgroundColor: "#f8f9fa",
  },
  cameraContainer: {
    alignItems: "center",
    marginBottom: 30,

  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    marginTop: 10,
  },
  subtitle: {
    fontSize: 18,
    color: "gray",
    marginTop: 5,
    marginBottom: 10,
  },
  image: {
    width: 300,
    height: 200,
    marginTop: 10,
    borderWidth: 1,
    borderColor: "#ddd",
  },
  webview: {
    width: 300,
    height: 200,
    marginTop: 10,
    borderWidth: 1,
    borderColor: 'black',
  },
  buttonContainer: {
    marginVertical: 10,
    width: '50%',
  },
});

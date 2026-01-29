import React from "react";
import {
  View,
  Text,
  Image,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  Switch,
} from "react-native";
import { Ionicons, MaterialIcons } from "@expo/vector-icons"; // Expo icons

const HomeScreen = () => {
  const [motionDetection, setMotionDetection] = React.useState(true);

  const motionEvents = [
    { id: "1", title: "Motion Detected at Front Door", time: "10:25 AM" },
    { id: "2", title: "Motion Detected in Living Room", time: "09:58 AM" },
  ];

  const recordedVideos = [
    {
      id: "1",
      title: "Motion Event - 10:25 AM",
      thumbnail: "https://via.placeholder.com/150",
    },
    {
      id: "2",
      title: "Motion Event - 09:58 AM",
      thumbnail: "https://via.placeholder.com/150",
    },
  ];

  return (
    <View style={styles.container}>
      {/* Header */}
      <Text style={styles.header}>Security Camera Dashboard</Text>

      {/* Live Camera Feed (Static Image for Demo) */}
      <View style={styles.cameraContainer}>
        <Image
          source={{ uri: "https://via.placeholder.com/400x200" }} // Dummy camera feed
          style={styles.cameraFeed}
        />
        <Text style={styles.recIndicator}>🔴 REC</Text>
      </View>

      {/* Motion Detection Alerts */}
      <Text style={styles.sectionTitle}>Recent Motion Alerts</Text>
      <FlatList
        data={motionEvents}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <View style={styles.motionItem}>
            <Ionicons name="alert-circle" size={24} color="red" />
            <View style={styles.motionText}>
              <Text style={styles.motionTitle}>{item.title}</Text>
              <Text style={styles.motionTime}>{item.time}</Text>
            </View>
          </View>
        )}
      />

      {/* Recorded Videos */}
      <Text style={styles.sectionTitle}>Recorded Videos</Text>
      <FlatList
        horizontal
        data={recordedVideos}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <TouchableOpacity style={styles.videoItem}>
            <Image
              source={{ uri: item.thumbnail }}
              style={styles.videoThumbnail}
            />
            <MaterialIcons
              name="play-circle-outline"
              size={50}
              color="white"
              style={styles.playIcon}
            />
            <Text style={styles.videoTitle}>{item.title}</Text>
          </TouchableOpacity>
        )}
      />

      {/* Quick Actions */}
      <View style={styles.quickActions}>
        <Text>Motion Detection</Text>
        <Switch value={motionDetection} onValueChange={setMotionDetection} />
        <Text style={styles.cameraStatus}>🟢 Camera Online</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, backgroundColor: "#121212" },
  header: {
    fontSize: 20,
    fontWeight: "bold",
    color: "white",
    textAlign: "center",
    marginBottom: 10,
  },
  cameraContainer: {
    position: "relative",
    alignItems: "center",
    marginBottom: 15,
  },
  cameraFeed: { width: "100%", height: 200, borderRadius: 10 },
  recIndicator: {
    position: "absolute",
    top: 10,
    right: 10,
    fontSize: 16,
    color: "red",
    fontWeight: "bold",
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "bold",
    color: "white",
    marginVertical: 10,
  },
  motionItem: {
    flexDirection: "row",
    alignItems: "center",
    padding: 10,
    backgroundColor: "#222",
    borderRadius: 8,
    marginBottom: 8,
  },
  motionText: { marginLeft: 10 },
  motionTitle: { fontSize: 14, fontWeight: "bold", color: "white" },
  motionTime: { fontSize: 12, color: "gray" },
  videoItem: { marginRight: 10, position: "relative" },
  videoThumbnail: { width: 120, height: 80, borderRadius: 10 },
  playIcon: { position: "absolute", top: "25%", left: "30%" },
  videoTitle: {
    textAlign: "center",
    color: "white",
    fontSize: 12,
    marginTop: 5,
  },
  quickActions: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginTop: 20,
    padding: 10,
    backgroundColor: "#222",
    borderRadius: 8,
  },
  cameraStatus: { fontWeight: "bold", color: "green" },
});

export default HomeScreen;
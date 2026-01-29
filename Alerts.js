import React, { useState } from "react";
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  TouchableOpacity,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";

export default function Alerts() {
  const [alerts, setAlerts] = useState([
    {
      id: "1",
      message: "Intruder detected at Front Door",
      time: "10 mins ago",
    },
    { id: "2", message: "Motion detected in Backyard", time: "30 mins ago" },
    { id: "3", message: "Intruder detected at Garage", time: "1 hour ago" },
  ]);

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Intruder Alerts</Text>
      {alerts.length === 0 ? (
        <Text style={styles.noAlerts}>No new alerts</Text>
      ) : (
        <FlatList
          data={alerts}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <View style={styles.alertCard}>
              <Ionicons name="alert-circle" size={24} color="red" />
              <View style={styles.alertTextContainer}>
                <Text style={styles.alertMessage}>{item.message}</Text>
                <Text style={styles.alertTime}>{item.time}</Text>
              </View>
            </View>
          )}
        />
      )}

      {/* Clear Alerts Button */}
      <TouchableOpacity
        style={styles.clearButton}
        onPress={() => setAlerts([])}
      >
        <Text style={styles.clearButtonText}>Clear All Alerts</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f8f9fa", padding: 10 },
  header: {
    fontSize: 24,
    fontWeight: "bold",
    textAlign: "center",
    marginBottom: 10,
  },
  noAlerts: { textAlign: "center", fontSize: 18, color: "gray", marginTop: 50 },

  alertCard: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#fff",
    padding: 10,
    marginVertical: 5,
    borderRadius: 8,
    elevation: 3,
  },
  alertTextContainer: { marginLeft: 10 },
  alertMessage: { fontSize: 16, fontWeight: "bold" },
  alertTime: { fontSize: 14, color: "gray" },

  clearButton: {
    backgroundColor: "red",
    padding: 12,
    borderRadius: 8,
    marginTop: 20,
    alignSelf: "center",
    width: "90%",
  },
  clearButtonText: { color: "#fff", fontWeight: "bold", textAlign: "center" },
});

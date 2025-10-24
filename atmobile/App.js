import React, { useState, useEffect, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  ScrollView,
  Alert,
  ActivityIndicator,
  Image,
  Platform,
  FlatList,
} from 'react-native';
// import { Camera, CameraType } from 'expo-camera';
import { CameraView, useCameraPermissions } from 'expo-camera';


import * as Location from 'expo-location';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Ionicons from 'react-native-vector-icons/Ionicons';

const API_URL = 'http://192.168.8.100:8000';
const Stack = createStackNavigator();
const Tab = createBottomTabNavigator();

// ============= Authentication Service =============
const AuthService = {
  async login(email, password) {
    const response = await fetch(`${API_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Login failed');
    await AsyncStorage.setItem('token', data.token);
    await AsyncStorage.setItem('user', JSON.stringify(data.employee));
    return data;
  },

  async logout() {
    await AsyncStorage.removeItem('token');
    await AsyncStorage.removeItem('user');
  },

  async getToken() {
    return await AsyncStorage.getItem('token');
  },

  async getUser() {
    const user = await AsyncStorage.getItem('user');
    return user ? JSON.parse(user) : null;
  },
};

// ============= API Service =============
const ApiService = {
  async checkIn(latitude, longitude, imageUri) {
    const token = await AuthService.getToken();
    const formData = new FormData();
    formData.append('latitude', latitude);
    formData.append('longitude', longitude);
    formData.append('face_image', {
      uri: imageUri,
      type: 'image/jpeg',
      name: 'face.jpg',
    });

    const response = await fetch(`${API_URL}/api/attendance/check-in`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
      },
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Check-in failed');
    return data;
  },

  async checkOut(latitude, longitude, imageUri) {
    const token = await AuthService.getToken();
    const formData = new FormData();
    formData.append('latitude', latitude);
    formData.append('longitude', longitude);
    formData.append('face_image', {
      uri: imageUri,
      type: 'image/jpeg',
      name: 'face.jpg',
    });

    const response = await fetch(`${API_URL}/api/attendance/check-out`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
      },
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Check-out failed');
    return data;
  },

  async getStatus() {
    const token = await AuthService.getToken();
    const response = await fetch(`${API_URL}/api/employee/status`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    const data = await response.json();
    if (!response.ok) throw new Error('Failed to fetch status');
    return data;
  },

  async getMyAttendance() {
    const token = await AuthService.getToken();
    const response = await fetch(`${API_URL}/api/employee/my-attendance`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    const data = await response.json();
    if (!response.ok) throw new Error('Failed to fetch attendance');
    return data.records;
  },

  async registerEmployee(employeeData, imageUri) {
    const token = await AuthService.getToken();
    const formData = new FormData();
    formData.append('employee_id', employeeData.employee_id);
    formData.append('name', employeeData.name);
    formData.append('email', employeeData.email);
    formData.append('password', employeeData.password);
    formData.append('role', employeeData.role);
    formData.append('face_image', {
      uri: imageUri,
      type: 'image/jpeg',
      name: 'face.jpg',
    });

    const response = await fetch(`${API_URL}/api/admin/register-employee`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
      },
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Registration failed');
    return data;
  },

  async getAttendance(date, employeeId) {
    const token = await AuthService.getToken();
    let url = `${API_URL}/api/admin/attendance?`;
    if (date) url += `date=${date}&`;
    if (employeeId) url += `employee_id=${employeeId}`;
    
    const response = await fetch(url, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    const data = await response.json();
    if (!response.ok) throw new Error('Failed to fetch attendance');
    return data.records;
  },

  async getEmployees() {
    const token = await AuthService.getToken();
    const response = await fetch(`${API_URL}/api/admin/employees`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    const data = await response.json();
    if (!response.ok) throw new Error('Failed to fetch employees');
    return data.employees;
  },
};

// ============= Login Screen =============
function LoginScreen({ navigation }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }

    setLoading(true);
    try {
      const data = await AuthService.login(email, password);
      if (data.employee.role === 'admin') {
        navigation.replace('AdminTabs');
      } else {
        navigation.replace('EmployeeTabs');
      }
    } catch (error) {
      Alert.alert('Login Failed', error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.loginContainer}>
        <Text style={styles.title}>Employee Attendance</Text>
        <Text style={styles.subtitle}>Facial Recognition System</Text>

        <TextInput
          style={styles.input}
          placeholder="Email"
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
          keyboardType="email-address"
        />

        <TextInput
          style={styles.input}
          placeholder="Password"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
        />

        <TouchableOpacity
          style={[styles.button, loading && styles.buttonDisabled]}
          onPress={handleLogin}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>Login</Text>
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
}

// ============= Camera Capture Component =============
function CameraCapture({ onCapture, onCancel }) {
  const [facing, setFacing] = useState('front'); // Use 'front' or 'back' as strings
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef(null);

  const takePicture = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.7,
        base64: false,
      });
      onCapture(photo.uri);
    }
  };

  if (!permission) {
    // Camera permissions are still loading
    return <View style={styles.container}><Text>Requesting camera permission...</Text></View>;
  }
  
  if (!permission.granted) {
    // Camera permissions not granted yet
    return (
      <View style={styles.container}>
        <Text>No access to camera</Text>
        <TouchableOpacity onPress={requestPermission}>
          <Text>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.cameraContainer}>
      <CameraView 
        style={styles.camera} 
        facing={facing}
        ref={cameraRef}
      >
        <View style={styles.cameraOverlay}>
          <View style={styles.faceGuide} />
        </View>
      </CameraView>
      <View style={styles.cameraButtons}>
        <TouchableOpacity style={styles.cameraButton} onPress={onCancel}>
          <Text style={styles.cameraButtonText}>Cancel</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.captureButton} onPress={takePicture}>
          <View style={styles.captureButtonInner} />
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.cameraButton}
          onPress={() => {
            setFacing(current => (current === 'back' ? 'front' : 'back'));
          }}
        >
          <Text style={styles.cameraButtonText}>Flip</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// ============= Employee Home Screen =============
function EmployeeHomeScreen() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [capturing, setCapturing] = useState(false);
  const [processing, setProcessing] = useState(false);

  useEffect(() => {
    loadStatus();
    const interval = setInterval(loadStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadStatus = async () => {
    try {
      const data = await ApiService.getStatus();
      setStatus(data);
    } catch (error) {
      Alert.alert('Error', error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCheckIn = async (imageUri) => {
    setCapturing(false);
    setProcessing(true);

    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        throw new Error('Location permission denied');
      }

      const location = await Location.getCurrentPositionAsync({});
      await ApiService.checkIn(
        location.coords.latitude,
        location.coords.longitude,
        imageUri
      );

      Alert.alert('Success', 'Checked in successfully!');
      loadStatus();
    } catch (error) {
      Alert.alert('Check-In Failed', error.message);
    } finally {
      setProcessing(false);
    }
  };

  const handleCheckOut = async (imageUri) => {
    setCapturing(false);
    setProcessing(true);

    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        throw new Error('Location permission denied');
      }

      const location = await Location.getCurrentPositionAsync({});
      await ApiService.checkOut(
        location.coords.latitude,
        location.coords.longitude,
        imageUri
      );

      Alert.alert('Success', 'Checked out successfully!');
      loadStatus();
    } catch (error) {
      Alert.alert('Check-Out Failed', error.message);
    } finally {
      setProcessing(false);
    }
  };

  if (loading) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#007AFF" />
      </View>
    );
  }

  if (capturing) {
    return (
      <CameraCapture
        onCapture={status?.checked_in ? handleCheckOut : handleCheckIn}
        onCancel={() => setCapturing(false)}
      />
    );
  }

  return (
    <ScrollView style={styles.container}>
      <View style={styles.content}>
        <View style={styles.statusCard}>
          <Text style={styles.statusTitle}>Today's Status</Text>
          <View style={styles.statusRow}>
            <Text style={styles.statusLabel}>Check-In:</Text>
            <Text style={[styles.statusValue, status?.checked_in && styles.statusActive]}>
              {status?.checked_in ? status.check_in_time?.substring(11, 16) : 'Not checked in'}
            </Text>
          </View>
          <View style={styles.statusRow}>
            <Text style={styles.statusLabel}>Check-Out:</Text>
            <Text style={[styles.statusValue, status?.checked_out && styles.statusActive]}>
              {status?.checked_out ? status.check_out_time?.substring(11, 16) : 'Not checked out'}
            </Text>
          </View>
        </View>

        {processing ? (
          <View style={styles.processingContainer}>
            <ActivityIndicator size="large" color="#007AFF" />
            <Text style={styles.processingText}>Processing...</Text>
          </View>
        ) : (
          <>
            {!status?.checked_in ? (
              <TouchableOpacity
                style={[styles.actionButton, styles.checkInButton]}
                onPress={() => setCapturing(true)}
              >
                <Text style={styles.actionButtonText}>Check In</Text>
              </TouchableOpacity>
            ) : !status?.checked_out ? (
              <TouchableOpacity
                style={[styles.actionButton, styles.checkOutButton]}
                onPress={() => setCapturing(true)}
              >
                <Text style={styles.actionButtonText}>Check Out</Text>
              </TouchableOpacity>
            ) : (
              <View style={styles.completedContainer}>
                <Text style={styles.completedText}>✓ Attendance completed for today</Text>
              </View>
            )}
          </>
        )}

        <View style={styles.infoCard}>
          <Text style={styles.infoTitle}>Instructions</Text>
          <Text style={styles.infoText}>• Ensure your face is clearly visible</Text>
          <Text style={styles.infoText}>• Allow location access when prompted</Text>
          <Text style={styles.infoText}>• Check in at the start of your shift</Text>
          <Text style={styles.infoText}>• Check out at the end of your shift</Text>
        </View>
      </View>
    </ScrollView>
  );
}

// ============= Employee History Screen =============
function EmployeeHistoryScreen() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadRecords();
  }, []);

  const loadRecords = async () => {
    try {
      const data = await ApiService.getMyAttendance();
      setRecords(data);
    } catch (error) {
      Alert.alert('Error', error.message);
    } finally {
      setLoading(false);
    }
  };

  const renderRecord = ({ item }) => (
    <View style={styles.recordCard}>
      <Text style={styles.recordDate}>{item.date}</Text>
      <View style={styles.recordRow}>
        <Text style={styles.recordLabel}>Check-In:</Text>
        <Text style={styles.recordValue}>
          {item.check_in_time ? item.check_in_time.substring(11, 16) : 'N/A'}
        </Text>
      </View>
      <View style={styles.recordRow}>
        <Text style={styles.recordLabel}>Check-Out:</Text>
        <Text style={styles.recordValue}>
          {item.check_out_time ? item.check_out_time.substring(11, 16) : 'N/A'}
        </Text>
      </View>
      {item.check_in_location && (
        <Text style={styles.recordLocation}>
          📍 {item.check_in_location.latitude.toFixed(6)}, {item.check_in_location.longitude.toFixed(6)}
        </Text>
      )}
    </View>
  );

  if (loading) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#007AFF" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <FlatList
        data={records}
        renderItem={renderRecord}
        keyExtractor={(item) => item.id.toString()}
        contentContainerStyle={styles.listContainer}
        ListEmptyComponent={
          <Text style={styles.emptyText}>No attendance records found</Text>
        }
      />
    </View>
  );
}

// ============= Admin Register Employee Screen =============
function AdminRegisterScreen() {
  const [employeeId, setEmployeeId] = useState('');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('employee');
  const [capturedImage, setCapturedImage] = useState(null);
  const [capturing, setCapturing] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleCapture = (uri) => {
    setCapturedImage(uri);
    setCapturing(false);
  };

  const handleRegister = async () => {
    if (!employeeId || !name || !email || !password || !capturedImage) {
      Alert.alert('Error', 'Please fill all fields and capture face image');
      return;
    }

    setLoading(true);
    try {
      await ApiService.registerEmployee(
        { employee_id: employeeId, name, email, password, role },
        capturedImage
      );
      Alert.alert('Success', 'Employee registered successfully!');
      setEmployeeId('');
      setName('');
      setEmail('');
      setPassword('');
      setCapturedImage(null);
    } catch (error) {
      Alert.alert('Registration Failed', error.message);
    } finally {
      setLoading(false);
    }
  };

  if (capturing) {
    return (
      <CameraCapture
        onCapture={handleCapture}
        onCancel={() => setCapturing(false)}
      />
    );
  }

  return (
    <ScrollView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.sectionTitle}>Register New Employee</Text>

        <TextInput
          style={styles.input}
          placeholder="Employee ID"
          value={employeeId}
          onChangeText={setEmployeeId}
        />

        <TextInput
          style={styles.input}
          placeholder="Full Name"
          value={name}
          onChangeText={setName}
        />

        <TextInput
          style={styles.input}
          placeholder="Email"
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
          keyboardType="email-address"
        />

        <TextInput
          style={styles.input}
          placeholder="Password"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
        />

        <View style={styles.roleContainer}>
          <Text style={styles.roleLabel}>Role:</Text>
          <TouchableOpacity
            style={[styles.roleButton, role === 'employee' && styles.roleButtonActive]}
            onPress={() => setRole('employee')}
          >
            <Text style={[styles.roleButtonText, role === 'employee' && styles.roleButtonTextActive]}>
              Employee
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.roleButton, role === 'admin' && styles.roleButtonActive]}
            onPress={() => setRole('admin')}
          >
            <Text style={[styles.roleButtonText, role === 'admin' && styles.roleButtonTextActive]}>
              Admin
            </Text>
          </TouchableOpacity>
        </View>

        {capturedImage ? (
          <View style={styles.imagePreviewContainer}>
            <Image source={{ uri: capturedImage }} style={styles.imagePreview} />
            <TouchableOpacity
              style={styles.retakeButton}
              onPress={() => setCapturing(true)}
            >
              <Text style={styles.retakeButtonText}>Retake Photo</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <TouchableOpacity
            style={styles.capturePhotoButton}
            onPress={() => setCapturing(true)}
          >
            <Text style={styles.capturePhotoButtonText}>📷 Capture Face Photo</Text>
          </TouchableOpacity>
        )}

        <TouchableOpacity
          style={[styles.button, loading && styles.buttonDisabled]}
          onPress={handleRegister}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>Register Employee</Text>
          )}
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

// ============= Admin Attendance Screen =============
function AdminAttendanceScreen() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);

  useEffect(() => {
    loadRecords();
  }, [selectedDate]);

  const loadRecords = async () => {
    setLoading(true);
    try {
      const data = await ApiService.getAttendance(selectedDate);
      setRecords(data);
    } catch (error) {
      Alert.alert('Error', error.message);
    } finally {
      setLoading(false);
    }
  };

  const openMap = (lat, lon) => {
    const url = Platform.select({
      ios: `maps:0,0?q=${lat},${lon}`,
      android: `geo:0,0?q=${lat},${lon}`,
    });
    Alert.alert('Location', `Lat: ${lat.toFixed(6)}\nLon: ${lon.toFixed(6)}`);
  };

  const renderRecord = ({ item }) => (
    <View style={styles.recordCard}>
      <Text style={styles.recordName}>{item.name}</Text>
      <Text style={styles.recordId}>ID: {item.employee_id}</Text>
      <View style={styles.recordRow}>
        <Text style={styles.recordLabel}>Check-In:</Text>
        <Text style={styles.recordValue}>
          {item.check_in_time ? item.check_in_time.substring(11, 16) : 'N/A'}
        </Text>
      </View>
      {item.check_in_location && (
        <TouchableOpacity
          onPress={() => openMap(item.check_in_location.latitude, item.check_in_location.longitude)}
        >
          <Text style={styles.recordLocationLink}>
            📍 View Check-In Location
          </Text>
        </TouchableOpacity>
      )}
      <View style={styles.recordRow}>
        <Text style={styles.recordLabel}>Check-Out:</Text>
        <Text style={styles.recordValue}>
          {item.check_out_time ? item.check_out_time.substring(11, 16) : 'N/A'}
        </Text>
      </View>
      {item.check_out_location && (
        <TouchableOpacity
          onPress={() => openMap(item.check_out_location.latitude, item.check_out_location.longitude)}
        >
          <Text style={styles.recordLocationLink}>
            📍 View Check-Out Location
          </Text>
        </TouchableOpacity>
      )}
    </View>
  );

  return (
    <View style={styles.container}>
      <View style={styles.dateSelector}>
        <Text style={styles.dateSelectorLabel}>Date: {selectedDate}</Text>
        <TouchableOpacity
          style={styles.todayButton}
          onPress={() => setSelectedDate(new Date().toISOString().split('T')[0])}
        >
          <Text style={styles.todayButtonText}>Today</Text>
        </TouchableOpacity>
      </View>
      {loading ? (
        <ActivityIndicator size="large" color="#007AFF" />
      ) : (
        <FlatList
          data={records}
          renderItem={renderRecord}
          keyExtractor={(item) => item.id.toString()}
          contentContainerStyle={styles.listContainer}
          ListEmptyComponent={
            <Text style={styles.emptyText}>No attendance records for this date</Text>
          }
        />
      )}
    </View>
  );
}

// ============= Admin Employees Screen =============
function AdminEmployeesScreen() {
  const [employees, setEmployees] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadEmployees();
  }, []);

  const loadEmployees = async () => {
    try {
      const data = await ApiService.getEmployees();
      setEmployees(data);
    } catch (error) {
      Alert.alert('Error', error.message);
    } finally {
      setLoading(false);
    }
  };

  const renderEmployee = ({ item }) => (
    <View style={styles.employeeCard}>
      <Text style={styles.employeeName}>{item.name}</Text>
      <Text style={styles.employeeDetail}>ID: {item.employee_id}</Text>
      <Text style={styles.employeeDetail}>Email: {item.email}</Text>
      <Text style={styles.employeeDetail}>Role: {item.role}</Text>
      <Text style={styles.employeeDate}>
        Registered: {item.created_at?.substring(0, 10)}
      </Text>
    </View>
  );

  if (loading) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#007AFF" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <FlatList
        data={employees}
        renderItem={renderEmployee}
        keyExtractor={(item) => item.employee_id}
        contentContainerStyle={styles.listContainer}
        ListEmptyComponent={
          <Text style={styles.emptyText}>No employees registered</Text>
        }
      />
    </View>
  );
}

// ============= Profile Screen =============
function ProfileScreen({ navigation }) {
  const [user, setUser] = useState(null);

  useEffect(() => {
    loadUser();
  }, []);

  const loadUser = async () => {
    const userData = await AuthService.getUser();
    setUser(userData);
  };

  const handleLogout = async () => {
    Alert.alert('Logout', 'Are you sure you want to logout?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Logout',
        onPress: async () => {
          await AuthService.logout();
          navigation.replace('Login');
        },
      },
    ]);
  };

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        <View style={styles.profileCard}>
          <Text style={styles.profileName}>{user?.name}</Text>
          <Text style={styles.profileDetail}>ID: {user?.id}</Text>
          <Text style={styles.profileDetail}>Email: {user?.email}</Text>
          <Text style={styles.profileDetail}>Role: {user?.role}</Text>
        </View>

        <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
          <Text style={styles.logoutButtonText}>Logout</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// ============= Tab Navigators =============
// function EmployeeTabs() {
//   return (
//     <Tab.Navigator
//       screenOptions={{
//         tabBarActiveTintColor: '#007AFF',
//         tabBarInactiveTintColor: 'gray',
//       }}
//     >
//       <Tab.Screen
//         name="Home"
//         component={EmployeeHomeScreen}
//         options={{ title: 'Attendance' }}
//       />
//       <Tab.Screen
//         name="History"
//         component={EmployeeHistoryScreen}
//         options={{ title: 'History' }}
//       />
//       <Tab.Screen
//         name="Profile"
//         component={ProfileScreen}
//         options={{ title: 'Profile' }}
//       />
//     </Tab.Navigator>
//   );
// }
function EmployeeTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarActiveTintColor: '#007AFF',
        tabBarInactiveTintColor: 'gray',
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;

          if (route.name === 'Home') {
            iconName = focused ? 'home' : 'home-outline';
          } else if (route.name === 'History') {
            iconName = focused ? 'time' : 'time-outline';
          } else if (route.name === 'Profile') {
            iconName = focused ? 'person' : 'person-outline';
          }

          return <Ionicons name={iconName} size={size} color={color} />;
        },
      })}
    >
      <Tab.Screen name="Home" component={EmployeeHomeScreen} options={{ title: 'Attendance' }} />
      <Tab.Screen name="History" component={EmployeeHistoryScreen} options={{ title: 'History' }} />
      <Tab.Screen name="Profile" component={ProfileScreen} options={{ title: 'Profile' }} />
    </Tab.Navigator>
  );
}


// function AdminTabs() {
//   return (
//     <Tab.Navigator
//       screenOptions={{
//         tabBarActiveTintColor: '#007AFF',
//         tabBarInactiveTintColor: 'gray',
//       }}
//     >
//       <Tab.Screen
//         name="Register"
//         component={AdminRegisterScreen}
//         options={{ title: 'Register' }}
//       />
//       <Tab.Screen
//         name="Attendance"
//         component={AdminAttendanceScreen}
//         options={{ title: 'Attendance' }}
//       />
//       <Tab.Screen
//         name="Employees"
//         component={AdminEmployeesScreen}
//         options={{ title: 'Employees' }}
//       />
//       <Tab.Screen
//         name="Profile"
//         component={ProfileScreen}
//         options={{ title: 'Profile' }}
//       />
//     </Tab.Navigator>
//   );
// }
function AdminTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarActiveTintColor: '#007AFF',
        tabBarInactiveTintColor: 'gray',
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;

          if (route.name === 'Register') {
            iconName = focused ? 'person-add' : 'person-add-outline';
          } else if (route.name === 'Attendance') {
            iconName = focused ? 'checkmark-done' : 'checkmark-done-outline';
          } else if (route.name === 'Employees') {
            iconName = focused ? 'people' : 'people-outline';
          } else if (route.name === 'Profile') {
            iconName = focused ? 'person' : 'person-outline';
          }

          return <Ionicons name={iconName} size={size} color={color} />;
        },
      })}
    >
      <Tab.Screen name="Register" component={AdminRegisterScreen} options={{ title: 'Register' }} />
      <Tab.Screen name="Attendance" component={AdminAttendanceScreen} options={{ title: 'Attendance' }} />
      <Tab.Screen name="Employees" component={AdminEmployeesScreen} options={{ title: 'Employees' }} />
      <Tab.Screen name="Profile" component={ProfileScreen} options={{ title: 'Profile' }} />
    </Tab.Navigator>
  );
}

// ============= Main App =============
export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        <Stack.Screen name="Login" component={LoginScreen} />
        <Stack.Screen name="EmployeeTabs" component={EmployeeTabs} />
        <Stack.Screen name="AdminTabs" component={AdminTabs} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

// ============= Styles =============
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  content: {
    padding: 20,
  },
  loginContainer: {
    flex: 1,
    justifyContent: 'center',
    padding: 30,
    backgroundColor: '#FFFFFF',
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#007AFF',
    marginBottom: 8,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    marginBottom: 40,
    textAlign: 'center',
  },
  input: {
    backgroundColor: '#F8F8F8',
    borderRadius: 10,
    padding: 15,
    marginBottom: 15,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  button: {
    backgroundColor: '#007AFF',
    borderRadius: 10,
    padding: 16,
    alignItems: 'center',
    marginTop: 10,
  },
  buttonDisabled: {
    backgroundColor: '#CCCCCC',
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
  },
  statusCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 15,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statusTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  statusRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  statusLabel: {
    fontSize: 16,
    color: '#666',
  },
  statusValue: {
    fontSize: 16,
    color: '#999',
  },
  statusActive: {
    color: '#34C759',
    fontWeight: '600',
  },
  actionButton: {
    borderRadius: 15,
    padding: 20,
    alignItems: 'center',
    marginBottom: 20,
  },
  checkInButton: {
    backgroundColor: '#34C759',
  },
  checkOutButton: {
    backgroundColor: '#FF3B30',
  },
  actionButtonText: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: 'bold',
  },
  processingContainer: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  processingText: {
    marginTop: 15,
    fontSize: 16,
    color: '#666',
  },
  completedContainer: {
    backgroundColor: '#E8F5E9',
    borderRadius: 15,
    padding: 20,
    alignItems: 'center',
    marginBottom: 20,
  },
  completedText: {
    fontSize: 18,
    color: '#34C759',
    fontWeight: '600',
  },
  infoCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 15,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  infoTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  infoText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 8,
  },
  cameraContainer: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  cameraOverlay: {
    flex: 1,
    backgroundColor: 'transparent',
    justifyContent: 'center',
    alignItems: 'center',
  },
  faceGuide: {
    width: 250,
    height: 300,
    borderWidth: 3,
    borderColor: '#FFFFFF',
    borderRadius: 150,
    backgroundColor: 'transparent',
  },
  cameraButtons: {
    flexDirection: 'row',
    backgroundColor: '#000',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingVertical: 20,
  },
  cameraButton: {
    padding: 15,
  },
  cameraButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  captureButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: '#FFFFFF',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 5,
    borderColor: '#007AFF',
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#007AFF',
  },
  listContainer: {
    padding: 15,
  },
  recordCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 15,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  recordDate: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#007AFF',
    marginBottom: 8,
  },
  recordName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  recordId: {
    fontSize: 14,
    color: '#666',
    marginBottom: 8,
  },
  recordRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 5,
  },
  recordLabel: {
    fontSize: 14,
    color: '#666',
  },
  recordValue: {
    fontSize: 14,
    color: '#333',
    fontWeight: '500',
  },
  recordLocation: {
    fontSize: 12,
    color: '#999',
    marginTop: 8,
  },
  recordLocationLink: {
    fontSize: 14,
    color: '#007AFF',
    marginTop: 5,
    textDecorationLine: 'underline',
  },
  emptyText: {
    textAlign: 'center',
    color: '#999',
    fontSize: 16,
    marginTop: 40,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 20,
  },
  roleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  roleLabel: {
    fontSize: 16,
    color: '#666',
    marginRight: 15,
  },
  roleButton: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#007AFF',
    marginRight: 10,
  },
  roleButtonActive: {
    backgroundColor: '#007AFF',
  },
  roleButtonText: {
    color: '#007AFF',
    fontSize: 14,
    fontWeight: '600',
  },
  roleButtonTextActive: {
    color: '#FFFFFF',
  },
  capturePhotoButton: {
    backgroundColor: '#007AFF',
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    marginBottom: 20,
  },
  capturePhotoButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  imagePreviewContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  imagePreview: {
    width: 200,
    height: 200,
    borderRadius: 100,
    marginBottom: 15,
  },
  retakeButton: {
    backgroundColor: '#FF9500',
    borderRadius: 8,
    paddingHorizontal: 20,
    paddingVertical: 10,
  },
  retakeButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
  dateSelector: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  dateSelectorLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  todayButton: {
    backgroundColor: '#007AFF',
    borderRadius: 8,
    paddingHorizontal: 15,
    paddingVertical: 8,
  },
  todayButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
  employeeCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 15,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  employeeName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 6,
  },
  employeeDetail: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  employeeDate: {
    fontSize: 12,
    color: '#999',
    marginTop: 6,
  },
  profileCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 15,
    padding: 25,
    marginBottom: 30,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  profileName: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
    textAlign: 'center',
  },
  profileDetail: {
    fontSize: 16,
    color: '#666',
    marginBottom: 8,
    textAlign: 'center',
  },
  logoutButton: {
    backgroundColor: '#FF3B30',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  logoutButtonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
  },
});
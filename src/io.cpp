#include "io.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <iomanip>
#include <cstring>
#include <unordered_map>
#include <filesystem>

// Global stable map for orbit_class strings
// assign incremental IDs as we encounter new classes
static std::unordered_map<std::string,int> orbitClassMap;
static int orbitClassCounter=0;


// function to read csv and populate vectors, with optional max_bodies limit
bool readCSV(const std::string& filename,
             std::vector<double>& masses,
             std::vector<Position>& positions,
             std::vector<Velocity>& velocities,
             std::vector<std::string>& names,
             std::vector<int>& orbit_classes,
             int max_bodies) { // added max_bodies parameter
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "error opening csv file: " << filename << "\n";
        return false;
    }

    std::string line;
    if (!std::getline(file, line)) {
        std::cerr << "empty file or error reading csv: " << filename << "\n";
        return false;
    }

    // process the header line
    std::istringstream headerStream(line);
    std::vector<std::string> headers;
    {
        std::string h;
        while (std::getline(headerStream, h, ',')) {
            headers.push_back(h);
        }
    }

    std::unordered_map<std::string, int> headerMap;
    for (size_t i = 0; i < headers.size(); i++) {
        headerMap[headers[i]] = (int)i;
    }

    // validate required headers
    std::vector<std::string> required = {"id", "name", "class", "mass", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"};
    for (auto &r : required) {
        if (headerMap.find(r) == headerMap.end()) {
            std::cerr << "missing required header: " << r << "\n";
            return false;
        }
    }

    masses.clear();
    positions.clear();
    velocities.clear();
    names.clear();
    orbit_classes.clear();

    int count = 0; // counter for the number of bodies read

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() < 10) continue; // skip malformed lines

        try {
            std::string nm = tokens.at(headerMap.at("name"));
            std::string cl = tokens.at(headerMap.at("class"));

            double m = std::stod(tokens.at(headerMap.at("mass")));
            double x = std::stod(tokens.at(headerMap.at("pos_x")));
            double y = std::stod(tokens.at(headerMap.at("pos_y")));
            double z = std::stod(tokens.at(headerMap.at("pos_z")));
            double vx = std::stod(tokens.at(headerMap.at("vel_x")));
            double vy = std::stod(tokens.at(headerMap.at("vel_y")));
            double vz = std::stod(tokens.at(headerMap.at("vel_z")));

            int oc;
            auto it = orbitClassMap.find(cl);
            if (it == orbitClassMap.end()) {
                oc = orbitClassCounter++;
                orbitClassMap[cl] = oc;
            } else {
                oc = it->second;
            }

            masses.push_back(m);
            positions.push_back({x, y, z});
            velocities.push_back({vx, vy, vz});
            names.push_back(nm);
            orbit_classes.push_back(oc);

            count++;
            // stop reading if max_bodies is reached
            if (max_bodies > 0 && count >= max_bodies) {
                break;
            }

        } catch (...) {
            // skip malformed line
        }
    }

    file.close();
    return true;
}

// function to save intermediate simulation state for visualization
void saveState(const std::string& vs_dir, int vs_counter,
               const std::vector<double>& masses,
               const std::vector<Position>& positions,
               const std::vector<Velocity>& velocities) {
    std::ostringstream vs_filename_stream;
    vs_filename_stream<<vs_dir<<"/output_"<<std::setw(5)<<std::setfill('0')<<vs_counter<<".csv";
    std::string vs_filename=vs_filename_stream.str();

    std::ofstream vs_file(vs_filename);
    if (!vs_file.is_open()) {
        std::cerr<<"Failed to open visualization file:"<<vs_filename<<"\n";
    } else {
        vs_file<<"id,mass,x,y,z,vx,vy,vz\n";
        for (size_t i=0;i<masses.size();i++){
            vs_file<<i<<","<<masses[i]<<","
                   <<positions[i].x<<","<<positions[i].y<<","<<positions[i].z<<","
                   <<velocities[i].vx<<","<<velocities[i].vy<<","<<velocities[i].vz<<"\n";
        }
        vs_file.close();
    }
}


// function to write a vtp file for visualization
void writeVTPFile(int rank, int vs_counter,
                  const std::vector<double> &local_masses,
                  const std::vector<Position> &local_positions,
                  const std::vector<Velocity> &local_velocities,
                  const std::vector<Acceleration> &local_accelerations,
                  const std::vector<std::string> &local_names,
                  const std::vector<int> &local_orbit_classes,
                  int body_id_offset,
                  double kinetic_energy,
                  double potential_energy,
                  double total_energy,
                  double virial_equilibrium,
                  const std::string &vs_dir) {
    using namespace tinyxml2;
    namespace fs = std::filesystem;

    int numPoints = static_cast<int>(local_masses.size());
    if (numPoints == 0) return;

    std::vector<int> body_ids(numPoints);
    for (int i = 0; i < numPoints; i++) {
        body_ids[i] = body_id_offset + i;
    }

    fs::path rank_dir = fs::path(vs_dir) / std::to_string(rank);
    fs::create_directories(rank_dir);
    std::string filename = (rank_dir / ("sim." + std::to_string(vs_counter) + ".vtp")).string();

    XMLDocument doc;
    XMLElement* vtkFile = doc.NewElement("VTKFile");
    vtkFile->SetAttribute("type", "PolyData");
    vtkFile->SetAttribute("version", "0.1");
    vtkFile->SetAttribute("byte_order", "LittleEndian");
    vtkFile->SetAttribute("header_type", "UInt64");
    doc.InsertFirstChild(vtkFile);

    XMLElement* polyData = doc.NewElement("PolyData");
    vtkFile->InsertEndChild(polyData);

    // Piece Element
    XMLElement* piece = doc.NewElement("Piece");
    piece->SetAttribute("NumberOfPoints", numPoints);
    piece->SetAttribute("NumberOfVerts", numPoints);
    polyData->InsertEndChild(piece);

    // Points
    {
        XMLElement* points = doc.NewElement("Points");
        piece->InsertEndChild(points);

        XMLElement* da = doc.NewElement("DataArray");
        da->SetAttribute("type", "Float64");
        da->SetAttribute("Name", "position");
        da->SetAttribute("NumberOfComponents", 3);
        da->SetAttribute("format", "ascii");
        std::ostringstream ss;
        for (int i = 0; i < numPoints; i++) {
            ss << local_positions[i].x << " " << local_positions[i].y << " " << local_positions[i].z << "\n";
        }
        da->InsertEndChild(doc.NewText(ss.str().c_str()));
        points->InsertEndChild(da);
    }

    // PointData
    {
        XMLElement* pd = doc.NewElement("PointData");
        piece->InsertEndChild(pd);

        auto writeIntArray = [&](const char* name, const std::vector<int>& arr) {
            XMLElement* da = doc.NewElement("DataArray");
            da->SetAttribute("type", "Int32");
            da->SetAttribute("Name", name);
            da->SetAttribute("NumberOfComponents", 1);
            da->SetAttribute("format", "ascii");
            std::ostringstream ss;
            for (auto v : arr) ss << v << "\n";
            da->InsertEndChild(doc.NewText(ss.str().c_str()));
            pd->InsertEndChild(da);
        };

        auto writeFloatArray = [&](const char* name, const std::vector<double>& arr, int comps = 1) {
            XMLElement* da = doc.NewElement("DataArray");
            da->SetAttribute("type", "Float64");
            da->SetAttribute("Name", name);
            da->SetAttribute("NumberOfComponents", comps);
            da->SetAttribute("format", "ascii");
            std::ostringstream ss;
            if (comps == 1) {
                for (auto v : arr) ss << v << "\n";
            } else {
                if (std::string(name) == "velocity") {
                    for (int i = 0; i < numPoints; i++) {
                        ss << local_velocities[i].vx << " " << local_velocities[i].vy << " " << local_velocities[i].vz << "\n";
                    }
                }
            }
            da->InsertEndChild(doc.NewText(ss.str().c_str()));
            pd->InsertEndChild(da);
        };

        // body_id
        writeIntArray("body_id", body_ids);

        // velocity
        writeFloatArray("velocity", {}, 3); // Empty vector since handled above

        // acceleration magnitude
        {
            std::vector<double> acc_magnitude(numPoints);
            for (int i = 0; i < numPoints; i++) {
                double ax = local_accelerations[i].ax;
                double ay = local_accelerations[i].ay;
                double az = local_accelerations[i].az;
                acc_magnitude[i] = std::sqrt(ax * ax + ay * ay + az * az);
            }
            writeFloatArray("acceleration", acc_magnitude, 1);
        }

        // mass
        writeFloatArray("mass", local_masses, 1);

        // name
        {
            XMLElement* da_name = doc.NewElement("DataArray");
            da_name->SetAttribute("type", "String");
            da_name->SetAttribute("Name", "name");
            da_name->SetAttribute("NumberOfComponents", 1);
            da_name->SetAttribute("format", "ascii");
            std::ostringstream ns;
            for (const auto &nm : local_names) {
                for (char c : nm) ns << static_cast<int>(c) << " ";
                ns << "0\n";
            }
            da_name->InsertEndChild(doc.NewText(ns.str().c_str()));
            pd->InsertEndChild(da_name);
        }

        // orbit_class
        writeIntArray("orbit_class", local_orbit_classes);
    }

    // FieldData
    {
        XMLElement* fieldData = doc.NewElement("FieldData");
        polyData->InsertEndChild(fieldData);

        auto writeScalar = [&](const char* name, double val) {
            XMLElement* da = doc.NewElement("DataArray");
            da->SetAttribute("type", "Float64");
            da->SetAttribute("Name", name);
            da->SetAttribute("NumberOfTuples", 1);
            da->SetAttribute("format", "ascii");
            std::ostringstream ss; ss << val;
            da->InsertEndChild(doc.NewText(ss.str().c_str()));
            fieldData->InsertEndChild(da);
        };

        writeScalar("kinetic energy", kinetic_energy);
        writeScalar("potential energy", potential_energy);
        writeScalar("total energy", total_energy);
        writeScalar("virial equilibrium", virial_equilibrium);
    }

    // Verts
    {
        XMLElement* verts = doc.NewElement("Verts");
        piece->InsertEndChild(verts);

        {
            XMLElement* da = doc.NewElement("DataArray");
            da->SetAttribute("type", "Int64");
            da->SetAttribute("Name", "offsets");
            std::ostringstream ss;
            for (int i = 1; i <= numPoints; i++) ss << i << " ";
            da->InsertEndChild(doc.NewText(ss.str().c_str()));
            verts->InsertEndChild(da);
        }

        {
            XMLElement* da = doc.NewElement("DataArray");
            da->SetAttribute("type", "Int64");
            da->SetAttribute("Name", "connectivity");
            std::ostringstream ss;
            for (int i = 0; i < numPoints; i++) ss << i << " ";
            da->InsertEndChild(doc.NewText(ss.str().c_str()));
            verts->InsertEndChild(da);
        }
    }

    doc.SaveFile(filename.c_str());
    doc.Clear();
}

 // function to update (or create) a pvd file that references all vtp files for the given timestep
void updatePVDFile(const std::string &pvdFilename,
                   int size,
                   int vs_counter,
                   double current_time,
                   const std::string &vs_dir) {
    using namespace tinyxml2;

    XMLDocument doc;
    XMLError e;

    XMLElement* vtkFile = nullptr;
    XMLElement* collection = nullptr;

    std::string pvdPath = (std::filesystem::path(vs_dir) / pvdFilename).string();

    if (vs_counter == 0) {
        // create a new .pvd file or overwrite existing one
        vtkFile = doc.NewElement("VTKFile");
        vtkFile->SetAttribute("type", "Collection");
        vtkFile->SetAttribute("version", "0.1");
        vtkFile->SetAttribute("byte_order", "LittleEndian");
        vtkFile->SetAttribute("compressor", "vtkZLibDataCompressor");
        doc.InsertFirstChild(vtkFile);

        // Create the Collection element
        collection = doc.NewElement("Collection");
        vtkFile->InsertEndChild(collection);
    }
    else {

        // attempt to load the existing .pvd file
        e = doc.LoadFile(pvdPath.c_str());

        if (e != XML_SUCCESS) {
            // if loading fails like file doesnt exist, create a new .pvd structure
            vtkFile = doc.NewElement("VTKFile");
            vtkFile->SetAttribute("type", "Collection");
            vtkFile->SetAttribute("version", "0.1");
            vtkFile->SetAttribute("byte_order", "LittleEndian");
            vtkFile->SetAttribute("compressor", "vtkZLibDataCompressor");
            doc.InsertFirstChild(vtkFile);

            collection = doc.NewElement("Collection");
            vtkFile->InsertEndChild(collection);
        }
        else {
            // locate the VTKFile element
            vtkFile = doc.FirstChildElement("VTKFile");
            if (!vtkFile) {
                // if VTKFile element is missing, recreate it
                doc.Clear();
                vtkFile = doc.NewElement("VTKFile");
                vtkFile->SetAttribute("type", "Collection");
                vtkFile->SetAttribute("version", "0.1");
                vtkFile->SetAttribute("byte_order", "LittleEndian");
                vtkFile->SetAttribute("compressor", "vtkZLibDataCompressor");
                doc.InsertFirstChild(vtkFile);

                collection = doc.NewElement("Collection");
                vtkFile->InsertEndChild(collection);
            }
            else {
                // locate the Collection element within VTKFile
                collection = vtkFile->FirstChildElement("Collection");
                if (!collection) {
                    // if Collection element is missing, create it
                    collection = doc.NewElement("Collection");
                    vtkFile->InsertEndChild(collection);
                }
            }
        }
    }

    // Add DataSet elements for each MPI rank**

    // Format the current_time with two decimal places
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << current_time;
    std::string t_str = oss.str();

    for (int r = 0; r < size; r++) {
        // create a new DataSet element
        XMLElement* dataSet = doc.NewElement("DataSet");
        dataSet->SetAttribute("timestep", t_str.c_str());
        // dataSet->SetAttribute("group", "");
        dataSet->SetAttribute("part", r);

        // construct the file reference path
        std::ostringstream fileRef;
        fileRef << r << "/sim." << vs_counter << ".vtp";
        dataSet->SetAttribute("file", fileRef.str().c_str());

        // append the DataSet to the Collection
        collection->InsertEndChild(dataSet);
    }

    // save the pvd

    XMLError saveResult = doc.SaveFile(pvdPath.c_str());
    if (saveResult != XML_SUCCESS) {
        std::cerr << "Error saving .pvd file: " << pvdPath << std::endl;
    }

    doc.Clear();
}

// Save final positions to CSV for the reference run (groundtruth)
void saveReferenceCSV(const std::string& dir, const std::vector<Position>& pos) {
    std::ofstream os(dir + "/final_ref.csv");
    os << "id,x,y,z\n";
    for (int i = 0; i < static_cast<int>(pos.size()); ++i) {
        os << i << "," << pos[i].x << "," << pos[i].y << "," << pos[i].z << "\n";
    }
}

// Load final reference positions from CSV
std::vector<Position> loadReferenceCSV(const std::string& dir, int num_bodies) {
    std::ifstream is(dir + "/final_ref.csv");
    if (!is.is_open()) {
        throw std::runtime_error("could not open reference CSV.");
    }

    std::string line;
    std::getline(is, line); // skip header

    std::vector<Position> pos(num_bodies);
    while (std::getline(is, line)) {
        std::istringstream ss(line);
        int id; char comma;
        double x, y, z;
        ss >> id >> comma >> x >> comma >> y >> comma >> z;
        pos[id] = {x, y, z};
    }
    return pos;
}



// Sum Euclidean distance between corresponding points in a and b
double computeDistanceSum(const std::vector<Position>& a,
                          const std::vector<Position>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double dx = a[i].x - b[i].x;
        double dy = a[i].y - b[i].y;
        double dz = a[i].z - b[i].z;
        sum += std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    return sum;
}

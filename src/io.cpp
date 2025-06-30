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
#include "tinyxml2.h" 


bool readCSV(
    const std::string& filename,
    std::vector<uint64_t>& ids, // New parameter
    std::vector<double>& masses,
    std::vector<Position>& positions,
    std::vector<Velocity>& velocities,
    int body_count) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    // Clear vectors to ensure they are empty before reading new data
    ids.clear();
    masses.clear();
    positions.clear();
    velocities.clear();

    std::string line;
    // Skip the header line of the CSV file
    std::getline(file, line);

    uint64_t current_id = 0; 

    while (std::getline(file, line)) {
        // Stop if the desired number of bodies has been read
        if (body_count > 0 && masses.size() >= static_cast<size_t>(body_count)) {
            break;
        }

        std::stringstream ss(line);
        std::string field;
        
        try {
            // Skip unused columns: id, name, class
            std::getline(ss, field, ','); // Skip id
            std::getline(ss, field, ','); // Skip name
            std::getline(ss, field, ','); // Skip class

            // Read the required data: mass, position, velocity
            double mass, px, py, pz, vx, vy, vz;
            
            std::getline(ss, field, ','); mass = std::stod(field);
            std::getline(ss, field, ','); px = std::stod(field);
            std::getline(ss, field, ','); py = std::stod(field);
            std::getline(ss, field, ','); pz = std::stod(field);
            std::getline(ss, field, ','); vx = std::stod(field);
            std::getline(ss, field, ','); vy = std::stod(field);
            std::getline(ss, field, ','); vz = std::stod(field);

            // Store the parsed values
            masses.push_back(mass);
            positions.push_back({px, py, pz});
            velocities.push_back({vx, vy, vz});

            ids.push_back(current_id++);

        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse line, skipping: \"" << line << "\"" << std::endl;
        }
    }

    return true;
}


/**
 * @brief Writes a snapshot file for one rank in VTP format.
 * Adapted from your original writeVTPFile.
 */
void writeSnapshot(int rank, int vs_counter,
                   const std::vector<uint64_t>& local_ids,      
                   const std::vector<double>& local_masses,
                   const std::vector<Position>& local_positions,
                   const std::vector<Velocity>& local_velocities,
                   const std::vector<Acceleration>& local_accelerations,
                   const std::string& vs_dir)
{
    using namespace tinyxml2;
    namespace fs = std::filesystem;
    (void)local_accelerations;
    int numPoints = static_cast<int>(local_masses.size());
    if (numPoints == 0) return;

    // Create a subdirectory for each rank, just like the original
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
        for (const auto& p : local_positions) {
            ss << p.x << " " << p.y << " " << p.z << "\n";
        }
        da->InsertEndChild(doc.NewText(ss.str().c_str()));
        points->InsertEndChild(da);
    }

    // PointData
    {
        XMLElement* pd = doc.NewElement("PointData");
        piece->InsertEndChild(pd);

        // --- body_id ---
        XMLElement* da_id = doc.NewElement("DataArray");
        da_id->SetAttribute("type", "UInt64");
        da_id->SetAttribute("Name", "body_id");
        da_id->SetAttribute("format", "ascii");
        std::ostringstream ss_id;
        for (auto v : local_ids) ss_id << v << "\n";
        da_id->InsertEndChild(doc.NewText(ss_id.str().c_str()));
        pd->InsertEndChild(da_id);

        // --- velocity ---
        XMLElement* da_vel = doc.NewElement("DataArray");
        da_vel->SetAttribute("type", "Float64");
        da_vel->SetAttribute("Name", "velocity");
        da_vel->SetAttribute("NumberOfComponents", 3);
        da_vel->SetAttribute("format", "ascii");
        std::ostringstream ss_vel;
        for (const auto& v : local_velocities) {
            ss_vel << v.x << " " << v.y << " " << v.z << "\n";
        }
        da_vel->InsertEndChild(doc.NewText(ss_vel.str().c_str()));
        pd->InsertEndChild(da_vel);

        // --- mass ---
        XMLElement* da_mass = doc.NewElement("DataArray");
        da_mass->SetAttribute("type", "Float64");
        da_mass->SetAttribute("Name", "mass");
        da_mass->SetAttribute("format", "ascii");
        std::ostringstream ss_mass;
        for (auto v : local_masses) ss_mass << v << "\n";
        da_mass->InsertEndChild(doc.NewText(ss_mass.str().c_str()));
        pd->InsertEndChild(da_mass);
    }

    // Verts
    {
        XMLElement* verts = doc.NewElement("Verts");
        piece->InsertEndChild(verts);
        // Offsets
        XMLElement* da_offsets = doc.NewElement("DataArray");
        da_offsets->SetAttribute("type", "Int64");
        da_offsets->SetAttribute("Name", "offsets");
        da_offsets->SetAttribute("format", "ascii");
        std::ostringstream ss_offsets;
        for (int i = 1; i <= numPoints; i++) ss_offsets << i << " ";
        da_offsets->InsertEndChild(doc.NewText(ss_offsets.str().c_str()));
        verts->InsertEndChild(da_offsets);
        // Connectivity
        XMLElement* da_conn = doc.NewElement("DataArray");
        da_conn->SetAttribute("type", "Int64");
        da_conn->SetAttribute("Name", "connectivity");
        da_conn->SetAttribute("format", "ascii");
        std::ostringstream ss_conn;
        for (int i = 0; i < numPoints; i++) ss_conn << i << " ";
        da_conn->InsertEndChild(doc.NewText(ss_conn.str().c_str()));
        verts->InsertEndChild(da_conn);
    }
    doc.SaveFile(filename.c_str());
}

void updatePVDFile(const cxxopts::ParseResult& args,
                   int size,
                   int vs_counter,
                   double current_time,
                   const std::string& vs_dir)
{
    using namespace tinyxml2;

    std::string input_stem = std::filesystem::path(args["file"].as<std::string>()).stem().string();
    std::string dt_str = args["dt"].as<std::string>();
    std::string tend_str = args["tend"].as<std::string>();
    double theta = args["theta"].as<double>();
    int nbodies = args["bodies"].as<int>();

    std::ostringstream oss_fname;
    oss_fname << input_stem
            << "_dt" << dt_str
            << "_tend" << tend_str
            << "_theta" << theta
            << "_bodies" << nbodies
            << ".pvd";
    std::string pvdFilename = oss_fname.str();
    std::string pvdPath = (std::filesystem::path(vs_dir) / pvdFilename).string();

    XMLDocument doc;
    XMLError e = doc.LoadFile(pvdPath.c_str());

    XMLElement* vtkFile = nullptr;
    XMLElement* collection = nullptr;

    if (e != XML_SUCCESS) {
        vtkFile = doc.NewElement("VTKFile");
        vtkFile->SetAttribute("type", "Collection");
        doc.InsertFirstChild(vtkFile);
        collection = doc.NewElement("Collection");
        vtkFile->InsertEndChild(collection);
    } else {
        vtkFile = doc.FirstChildElement("VTKFile");
        collection = vtkFile ? vtkFile->FirstChildElement("Collection") : nullptr;
        if (!collection) {
            doc.Clear();
            vtkFile = doc.NewElement("VTKFile");
            doc.InsertFirstChild(vtkFile);
            collection = doc.NewElement("Collection");
            vtkFile->InsertEndChild(collection);
        }
    }

    for (int r = 0; r < size; r++) {
        XMLElement* dataSet = doc.NewElement("DataSet");
        dataSet->SetAttribute("timestep", current_time);
        dataSet->SetAttribute("part", r);
        std::ostringstream fileRef;
        fileRef << r << "/sim." << vs_counter << ".vtp";
        dataSet->SetAttribute("file", fileRef.str().c_str());
        collection->InsertEndChild(dataSet);
    }
    doc.SaveFile(pvdPath.c_str());
}



// bool readCSV(
//     const std::string& filename,
//     std::vector<uint64_t>& ids,
//     std::vector<double>& masses,
//     std::vector<Position>& positions,
//     std::vector<Velocity>& velocities,
//     int body_count)
// {
//     std::ifstream file(filename);
//     if (!file.is_open()) {
//         std::cerr << "Error: Could not open file " << filename << std::endl;
//         return false;
//     }

//     // 1. Read the header line to map column names to indices
//     std::string header_line;
//     if (!std::getline(file, header_line)) {
//         std::cerr << "Error: CSV file is empty or cannot be read." << std::endl;
//         return false;
//     }
//     std::stringstream header_ss(header_line);
//     std::string header;
//     std::unordered_map<std::string, int> column_map;
//     int col_idx = 0;
//     while (std::getline(header_ss, header, ',')) {
//         // Trim whitespace/carriage returns
//         header.erase(header.find_last_not_of(" \n\r\t")+1);
//         column_map[header] = col_idx++;
//     }

//     // 2. Validate that all required headers are present
//     std::vector<std::string> required_headers = {
//         "mass", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"
//     };
//     for (const auto& req : required_headers) {
//         if (column_map.find(req) == column_map.end()) {
//             std::cerr << "Error: Missing required header in CSV file: " << req << std::endl;
//             return false;
//         }
//     }

//     // Clear output vectors
//     ids.clear();
//     masses.clear();
//     positions.clear();
//     velocities.clear();

//     // 3. Read data rows using the header map
//     std::string line;
//     uint64_t current_id = 0;
//     int bodies_read = 0;
//     while (std::getline(file, line)) {
//         if (body_count > 0 && bodies_read >= body_count) {
//             break;
//         }

//         std::stringstream line_ss(line);
//         std::string field;
//         std::vector<std::string> fields;
//         while (std::getline(line_ss, field, ',')) {
//             fields.push_back(field);
//         }

//         if (fields.size() < column_map.size()) continue; // Skip malformed lines

//         try {
//             // Access data using the map, not by fixed order
//             double m  = std::stod(fields.at(column_map.at("mass")));
//             double px = std::stod(fields.at(column_map.at("pos_x")));
//             double py = std::stod(fields.at(column_map.at("pos_y")));
//             double pz = std::stod(fields.at(column_map.at("pos_z")));
//             double vx = std::stod(fields.at(column_map.at("vel_x")));
//             double vy = std::stod(fields.at(column_map.at("vel_y")));
//             double vz = std::stod(fields.at(column_map.at("vel_z")));

//             // Store the parsed values
//             masses.push_back(m);
//             positions.push_back({px, py, pz});
//             velocities.push_back({vx, vy, vz});

//             // Generate a persistent ID
//             ids.push_back(current_id++);
//             bodies_read++;

//         } catch (const std::exception& e) {
//             std::cerr << "Warning: Could not parse line, skipping: \"" << line << "\"\n";
//         }
//     }
//     return true;
// }

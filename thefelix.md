// ----------- Color Parameters -----------
helix_color = "DeepSkyBlue";
post_color = "SlateGray";
node_color = "Red";
spoke_color = "Gold";

// ----------- Randomization -----------
random_seed = 42069; 

// ----------- Helix Parameters -----------
top_radius = 33;
bottom_radius = 0.001;
height = 33;
turns = 33;
wire_thickness = .3;
segments_per_turn = 33;

// ----------- Node and Spoke Parameters -----------
add_nodes_and_spokes = true;
number_of_nodes = 133;
node_diameter = .3;
spoke_thickness = 0.03;


// ----------- Model Generation -----------

node_start_times = rands(0, 1, number_of_nodes, random_seed);

generate_helix();

if (add_nodes_and_spokes) {
    generate_nodes_and_spokes();
}

// -- Module to generate the main helix body --
module generate_helix() {
    // NOTE: This now uses hull() instead of BOSL2's path_sweep
    color(helix_color) {
        total_steps = turns * segments_per_turn;
        for (i = [0 : total_steps - 1]) {
            p1 = get_position(i, turns, segments_per_turn, height, top_radius, bottom_radius);
            p2 = get_position(i + 1, turns, segments_per_turn, height, top_radius, bottom_radius);
            hull() {
                translate(p1) sphere(d=wire_thickness, $fn=8);
                translate(p2) sphere(d=wire_thickness, $fn=8);
            }
        }
    }
}

// ----------- Module for Nodes and Spokes -----------
module generate_nodes_and_spokes() {
    color(post_color)
        cylinder(h = height, d = node_diameter, center = false, $fn=32);

    for (i = [0 : number_of_nodes - 1]) {
        start_time = node_start_times[i];
        
        if ($t >= start_time) {
            path_fraction = $t - start_time;
            
            node_z = height * (1 - path_fraction);
            node_angle = turns * 360 * (1 - path_fraction);
            
            node_radius = bottom_radius * pow(top_radius / bottom_radius, node_z / height);
            
            node_pos = [
                node_radius * cos(node_angle),
                node_radius * sin(node_angle),
                node_z
            ];

            spoke_anchor_point = [0, 0, node_z];
            
            color(node_color)
                translate(node_pos)
                    sphere(d = node_diameter, $fn = 24);
            
            // CHANGED: Using hull() for the spokes. No library needed.
            color(spoke_color)
                hull() {
                    translate(spoke_anchor_point)
                        sphere(d=spoke_thickness, $fn=8);
                    translate(node_pos)
                        sphere(d=spoke_thickness, $fn=8);
                }
        }
    }
}

// -- Helper Functions --

function get_position(step, p_turns, p_segs, p_h, p_t_rad, p_b_rad) = let(
    total_steps = p_turns * p_segs,
    angle = step / total_steps * p_turns * 360,
    z = step / total_steps * p_h,
    r = p_b_rad * pow(p_t_rad / p_b_rad, z / p_h),
    x = r * cos(angle),
    y = r * sin(angle)
) [x, y, z];
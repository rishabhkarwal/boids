import pygame, sys
import numpy as np
import moderngl
import math

WIDTH, HEIGHT = 800, 800
fps = 120

N = 1_000 # number of agents

# colours
background_colour = (40, 44, 52)
agent_colour = (58, 124, 165)
outline_colour = (129, 195, 215)

agent_size = 4.0

# behavioural hyperparameters
separation = 7.0 ## steer to avoid crowding local flockmates
alignment = 1.3 ## steer towards the average heading of local flockmates
cohesion = 5.0 ## steer to move towards the average position of local flockmates

# sensory hyperparameters
avoidance = 15.0 ## distance to trigger wall/agent avoidance
object_force = 1.4 ## how strong object avoidance force is
vision = 50.0 ## how far an agent can see
fov_degrees = 220.0

max_speed = 1.5 
max_force = 0.1

pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode([WIDTH, HEIGHT], pygame.OPENGL | pygame.DOUBLEBUF | pygame.NOFRAME)
ctx = moderngl.create_context()

# clear cell counts
## resets the count of agents in each spatial grid cell to zero at the start of the frame
clear_counts_source = """
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 1) buffer CellCounts { int cell_counts[]; };
uniform int num_cells;

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id < num_cells) {
        cell_counts[id] = 0;
    }
}
"""

# count agents per cell
## determines which grid cell each agent is currently in and increments that cell's counter
count_agents_source = """
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer AgentsIn { vec4 agents_in[]; };
layout(std430, binding = 1) buffer CellCounts { int cell_counts[]; };

uniform float cell_size;
uniform int grid_width;
uniform int grid_height;
uniform int num_agents;

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= num_agents) return;

    vec2 pos = agents_in[id].xy;
    // calculate 2d grid index
    int cx = clamp(int(pos.x / cell_size), 0, grid_width - 1);
    int cy = clamp(int(pos.y / cell_size), 0, grid_height - 1);
    
    // atomically add 1 to the 1d array representing the 2d grid to prevent race conditions
    atomicAdd(cell_counts[cy * grid_width + cx], 1);
}
"""

# prefix sum
## calculates the starting index for each cell's agents in the final sorted array
## runs on a single thread to accumulate totals sequentially
prefix_sum_source = """
#version 430
layout(local_size_x = 1) in; 
layout(std430, binding = 1) buffer CellCounts { int cell_counts[]; };
layout(std430, binding = 2) buffer CellStarts { int cell_starts[]; };
uniform int num_cells;

void main() {
    int accum = 0;
    for(int i = 0; i < num_cells; i++) {
        cell_starts[i] = accum;
        accum += cell_counts[i];
        cell_counts[i] = 0; // reset count for the scatter phase
    }
}
"""

# scatter
## moves agents into a sorted array based on their spatial grid cell
## this ensures agents in the same cell are contiguous in memory for fast neighbour lookups
scatter_agents_source = """
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer AgentsIn { vec4 agents_in[]; };
layout(std430, binding = 1) buffer CellCounts { int cell_counts[]; };
layout(std430, binding = 2) buffer CellStarts { int cell_starts[]; };
layout(std430, binding = 3) buffer AgentsOut { vec4 agents_out[]; };

uniform float cell_size;
uniform int grid_width;
uniform int grid_height;
uniform int num_agents;

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= num_agents) return;

    vec4 bdata = agents_in[id];
    int cx = clamp(int(bdata.x / cell_size), 0, grid_width - 1);
    int cy = clamp(int(bdata.y / cell_size), 0, grid_height - 1);
    int cell_index = cy * grid_width + cx;
    
    // find exact slot for this agent in the sorted array
    int offset = atomicAdd(cell_counts[cell_index], 1);
    agents_out[cell_starts[cell_index] + offset] = bdata;
}
"""

# flocking
## calculates the new position and velocity for each agent based on its neighbours
compute_shader_source = """
#version 430
layout(local_size_x = 32) in;

layout(std430, binding = 0) buffer AgentsNext { vec4 agents_next[]; };
layout(std430, binding = 1) buffer CellCounts { int cell_counts[]; };
layout(std430, binding = 2) buffer CellStarts { int cell_starts[]; };
layout(std430, binding = 3) buffer AgentsSorted { vec4 agents_sorted[]; };

uniform float separation_weight, alignment_weight, cohesion_weight;
uniform float object_force, avoidance_sq, vision_sq, fov, max_speed, max_force, width, height;
uniform int num_agents, grid_width, grid_height;
uniform float cell_size;
uniform float agent_size;

const float PI = 3.14159265359;

// helper function to normalise vectors safely, avoiding dbz
vec2 normalise(vec2 v) {
    float m = length(v);
    return m > 0.0001 ? v / m : v;
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= num_agents) return;

    vec4 me = agents_sorted[id];
    vec2 pos = me.xy;
    vec2 vel = me.zw;
    float heading = atan(vel.y, vel.x);

    // accumulators for the three boids rules
    vec2 _sep = vec2(0.0), _alg = vec2(0.0), _coh = vec2(0.0);
    float c_sep = 0.0, c_alg = 0.0, c_coh = 0.0;

    int cx = clamp(int(pos.x / cell_size), 0, grid_width - 1);
    int cy = clamp(int(pos.y / cell_size), 0, grid_height - 1);

    // loop over the 3x3 grid of surrounding cells to find neighbours
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = cx + dx;
            int ny = cy + dy;
            
            // check if neighbour cell is within simulation bounds
            float valid_cell = step(0, nx) * step(nx, grid_width - 1) * step(0, ny) * step(ny, grid_height - 1);
            
            if (valid_cell > 0.5) {
                int cell_idx = ny * grid_width + nx;
                int start = cell_starts[cell_idx];
                int count = cell_counts[cell_idx];

                // iterate through every agent in this specific cell
                for (int i = 0; i < count; i++) {
                    int target = start + i;
                    if (target == int(id)) continue; // skip self

                    vec4 other = agents_sorted[target];
                    vec2 dir = other.xy - pos;
                    float d_sq = dot(dir, dir);
                    
                    // only process agents within visual range
                    float mask = step(0.0001, d_sq) * step(d_sq, vision_sq);
                    
                    // field of view check
                    float angle = mod(atan(dir.y, dir.x) - heading + PI, 2.0 * PI) - PI;
                    float fov_mask = mask * step(abs(angle), fov * 0.5);
                    
                    // separation: steer away from very close neighbours
                    float avoid_mask = mask * step(d_sq, avoidance_sq);
                    _sep += (-(dir / max(sqrt(d_sq), 0.1)) * avoid_mask);
                    c_sep += avoid_mask;

                    // alignment: match velocity of neighbours
                    _alg += (other.zw * fov_mask);
                    _coh += (other.xy * fov_mask);

                    c_alg += fov_mask;
                    c_coh += fov_mask;
                }
            }
        }
    }

    vec2 force = vec2(0.0);
    // apply weights to the accumulated vectors
    force += (c_sep > 0.0) ? normalise(_sep) * separation_weight : vec2(0.0);
    force += (c_alg > 0.0) ? normalise((_alg / max(c_alg, 1.0)) - vel) * alignment_weight : vec2(0.0);
    force += (c_coh > 0.0) ? normalise((_coh / max(c_coh, 1.0)) - pos) * cohesion_weight : vec2(0.0);

    // wall avoidance logic
    vec2 wall_f = vec2(0.0);
    wall_f.x += step(pos.x, avoidance_sq) * (1.0 / max(pos.x, 0.1));
    wall_f.x -= step(width - pos.x, avoidance_sq) * (1.0 / max(width - pos.x, 0.1));
    wall_f.y += step(pos.y, avoidance_sq) * (1.0 / max(pos.y, 0.1));
    wall_f.y -= step(height - pos.y, avoidance_sq) * (1.0 / max(height - pos.y, 0.1));
    force += (length(wall_f) > 0.0) ? normalise(wall_f) * object_force : vec2(0.0);

    // clamp steering force
    float f_len = length(force);
    force = (f_len > max_force) ? (force / f_len) * max_force : force;
    
    // update velocity and clamp speed
    vel += force;
    float s_len = length(vel);
    vel = (s_len > max_speed) ? (vel / s_len) * max_speed : vel;
    pos += vel;

    // hard boundary constraints
    if (pos.x <= agent_size) { pos.x = agent_size; }
    else if (pos.x >= width - agent_size) { pos.x = width - agent_size; }
    if (pos.y <= agent_size) { pos.y = agent_size; }
    else if (pos.y >= height - agent_size) { pos.y = height - agent_size; }

    agents_next[id] = vec4(pos, vel);
}
"""

# render
## used strictly for drawing the triangles to the screen
vertex_shader_source = """
#version 430
in vec2 in_vert; 
in vec4 in_agent_data; 
uniform float width, height, size;

void main() {
    // calculate rotation matrix based on velocity vector
    float heading = atan(in_agent_data.w, in_agent_data.z);
    mat2 rot = mat2(cos(heading), sin(heading), -sin(heading), cos(heading));

    // rotate and scale the base triangle
    vec2 final_pos = in_agent_data.xy + (rot * (in_vert * size));

    // convert screen coordinates to opengl normalised device coordinates (-1 to 1)
    gl_Position = vec4((final_pos.x / width) * 2.0 - 1.0, 1.0 - (final_pos.y / height) * 2.0, 0.0, 1.0);
}
"""

fragment_shader_source = """
#version 430
out vec4 f_color;
uniform vec3 render_colour;
void main() { f_color = vec4(render_colour, 1.0); }
"""

def main():
    # compile all shaders
    clear_prog = ctx.compute_shader(clear_counts_source)
    count_prog = ctx.compute_shader(count_agents_source)
    prefix_prog = ctx.compute_shader(prefix_sum_source)
    scatter_prog = ctx.compute_shader(scatter_agents_source)
    compute_prog = ctx.compute_shader(compute_shader_source)
    render_prog = ctx.program(vertex_shader=vertex_shader_source, fragment_shader=fragment_shader_source)

    # normalise colours from 0-255 to 0.0-1.0 for opengl
    bg_norm = tuple(c / 255.0 for c in background_colour)
    outline_norm = tuple(c / 255.0 for c in outline_colour)

    # determine optimal grid size based on maximum vision/avoidance distances
    cell_size = max(vision, avoidance)
    grid_width = math.ceil(WIDTH / cell_size)
    grid_height = math.ceil(HEIGHT / cell_size)
    num_cells = grid_width * grid_height

    # pass generic uniform data to all relevant shaders
    for prog in [clear_prog, count_prog, prefix_prog, scatter_prog, compute_prog]:
        if 'num_cells' in prog: prog['num_cells'].value = num_cells
        if 'cell_size' in prog: prog['cell_size'].value = cell_size
        if 'grid_width' in prog: prog['grid_width'].value = grid_width
        if 'grid_height' in prog: prog['grid_height'].value = grid_height
        if 'num_agents' in prog: prog['num_agents'].value = N

    # pass specific hyperparameters to the flocking shader
    compute_prog['separation_weight'].value = separation
    compute_prog['alignment_weight'].value = alignment
    compute_prog['cohesion_weight'].value = cohesion
    compute_prog['object_force'].value = object_force
    compute_prog['avoidance_sq'].value = avoidance * avoidance
    compute_prog['vision_sq'].value = vision * vision
    compute_prog['fov'].value = np.radians(fov_degrees)
    compute_prog['max_speed'].value = max_speed
    compute_prog['max_force'].value = max_force
    compute_prog['width'].value = float(WIDTH)
    compute_prog['height'].value = float(HEIGHT)
    compute_prog['agent_size'].value = agent_size

    # pass rendering data
    render_prog['width'].value = float(WIDTH)
    render_prog['height'].value = float(HEIGHT)
    render_prog['size'].value = agent_size
    render_prog['render_colour'].value = outline_norm

    # initialise agent data on the cpu using numpy
    # each agent holds 4 floats: [x, y, dx, dy]
    agent_data = np.zeros((N, 4), dtype='f4')
    agent_data[:, 0] = np.random.rand(N) * WIDTH
    agent_data[:, 1] = np.random.rand(N) * HEIGHT
    agent_data[:, 2:4] = (np.random.rand(N, 2) - 0.5) * max_speed

    # create gpu buffers (ssbos)
    agents_in_buffer = ctx.buffer(agent_data.tobytes())
    agents_out_buffer = ctx.buffer(reserve=N * 16) # 16 bytes per agent (4 floats * 4 bytes)
    cell_counts_buffer = ctx.buffer(reserve=num_cells * 4) # 4 bytes per integer
    cell_starts_buffer = ctx.buffer(reserve=num_cells * 4)

    # bind buffers to specific bindings in the compute shaders
    agents_in_buffer.bind_to_storage_buffer(0)
    cell_counts_buffer.bind_to_storage_buffer(1)
    cell_starts_buffer.bind_to_storage_buffer(2)
    agents_out_buffer.bind_to_storage_buffer(3)

    # define base geometry (a simple triangle pointing right)
    vbo_geom = ctx.buffer(np.array([[1, 0], [-1, 0.6], [-1, -0.6]], dtype='f4').tobytes())
    # vertex array object tying the geometry and instance data together
    vao = ctx.vertex_array(render_prog, [(vbo_geom, '2f', 'in_vert'), (agents_in_buffer, '4f /i', 'in_agent_data')])

    # calculate dispatch group dimensions for compute shaders
    group_size = 32
    agent_groups = int(np.ceil(N / group_size))
    cell_groups = int(np.ceil(num_cells / 256))

    while 1:
        for event in pygame.event.get():
            if (event.type == pygame.QUIT) or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit()

        # clear grid data
        clear_prog.run(group_x=cell_groups)
        ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)

        # assign agents to cells and count them
        count_prog.run(group_x=agent_groups)
        ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)

        # run prefix sum to get memory offsets for scattering
        prefix_prog.run(group_x=1)
        ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)

        # scatter agents into a contiguous sorted array
        scatter_prog.run(group_x=agent_groups)
        ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)

        # run main flocking logic using the sorted array
        compute_prog.run(group_x=agent_groups)
        ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)

        # rendering phase
        ctx.clear(*bg_norm)
        vao.render(mode=moderngl.TRIANGLES, instances=N)

        pygame.display.flip()
        clock.tick(fps)

if __name__ == '__main__':
    main()
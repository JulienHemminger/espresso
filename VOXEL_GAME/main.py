from collections import deque

from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController

# Initialize Ursina
app = Ursina()

# Enable basic shadows
import ursina.shaders

Entity.default_shader = ursina.shaders.lit_with_shadows_shader

# Hide default UI
window.fps_counter.enabled = False
window.exit_button.enabled = False

# Game configuration
BLOCK_SCALE = 0.125
STRUCTURE_MAX_SIZE = 10  # Maximum size of a structure that can be destroyed at once

active_block_type = 0
block_types = ["gravel", "sand"]
blocks_in_world = []
disjoint_lines = []
is_paused = False


class Block(Button):
    def __init__(self, position=(0, 0, 0), block_type="gravel"):
        block_color = color.dark_gray if block_type == "gravel" else color.yellow

        super().__init__(
            parent=scene,
            position=position,
            model="cube",
            scale=BLOCK_SCALE,
            origin_y=0.5,
            texture=None,
            color=block_color,
            highlight_color=block_color.tint(0.15),
        )
        self.block_type = block_type

        self.glued_faces = set()

        blocks_in_world.append(self)
        invoke(update_disjoint_lines, delay=0.02)

    def input(self, key):
        global active_block_type, is_paused
        if is_paused:
            return

        if self.hovered:
            # 1. Left-Click logic (Structure Destruction vs. Ungluing)
            if key == "left mouse down":
                if held_keys["shift"]:
                    # Shift + Left-Click: Unglue targeted face
                    print("TRIGGER UNGLUE FACE")
                    self.unglue_targeted_face()
                else:
                    # Left-Click: Destroy the entire connected structure
                    self.destroy_structure()

            # 2. Right-Click logic (Placement & Gluing)
            elif key == "right mouse down":
                normal = mouse.normal  # Direction of the face we clicked
                new_position = self.position + (normal * BLOCK_SCALE)

                # Check if Ctrl is held for gluing
                is_glued = held_keys["control"]

                new_block = Block(
                    position=new_position, block_type=block_types[active_block_type]
                )

                if is_glued:
                    # Glue both blocks to each other along the shared boundary
                    normal_tuple = (int(normal.x), int(normal.y), int(normal.z))
                    opposite_normal = (
                        -normal_tuple[0],
                        -normal_tuple[1],
                        -normal_tuple[2],
                    )

                    self.glued_faces.add(normal_tuple)
                    new_block.glued_faces.add(opposite_normal)

                invoke(update_disjoint_lines, delay=0.02)

    def destroy_structure(self):
        """Finds all glued blocks forming a structure using BFS and destroys them if within threshold."""
        structure = get_connected_structure(self)

        if len(structure) > STRUCTURE_MAX_SIZE:
            notify_text(
                f"Structure too big to destroy! ({len(structure)}/{STRUCTURE_MAX_SIZE} blocks)"
            )
            return

        for b in structure:
            if b in blocks_in_world:
                blocks_in_world.remove(b)
            destroy(b)

        notify_text(f"Destroyed structure of {len(structure)} blocks.")
        invoke(update_disjoint_lines, delay=0.02)

    def unglue_targeted_face(self):
        """Unglues the targeted block face from its adjacent neighbor."""
        normal = mouse.normal * -1.0
        normal_tuple = (int(normal.x), int(normal.y), int(normal.z))

        if normal_tuple in self.glued_faces:
            self.glued_faces.remove(normal_tuple)

            # Find the neighboring block in that direction to unglue its shared face as well
            neighbor_pos = self.position + (normal * BLOCK_SCALE)

            neighbor = find_block_at(neighbor_pos)
            if neighbor:
                opposite_normal = (-normal_tuple[0], -normal_tuple[1], -normal_tuple[2])
                if opposite_normal in neighbor.glued_faces:
                    neighbor.glued_faces.remove(opposite_normal)

            invoke(update_disjoint_lines, delay=0.02)

    def update(self):
        if is_paused:
            return

        # If a block is glued in ANY direction, or sits on the hard-coded y=0 floor, it does not fall
        if len(self.glued_faces) > 0 or self.y <= 0.001:
            return

        # Gravity check directly below
        target_below = self.position + Vec3(0, -BLOCK_SCALE, 0)

        is_blocked = False
        for b in blocks_in_world:
            if b != self and distance(b.position, target_below) < (BLOCK_SCALE * 0.1):
                is_blocked = True
                break

        if not is_blocked:
            self.y -= BLOCK_SCALE
            invoke(update_disjoint_lines, delay=0.01)


# --- Helper Utilities ---


def find_block_at(position):
    """Finds a block at a given coordinate with a floating point margin of error."""
    for b in blocks_in_world:
        if distance(b.position, position) < (BLOCK_SCALE * 0.1):
            return b
    return None


def get_connected_structure(start_block):
    """Performs a Breadth-First Search to return all blocks structurally glued together."""
    visited = set([start_block])
    queue = deque([start_block])

    while queue:
        current = queue.popleft()

        # Traverse only through explicitly glued directions
        for direction in current.glued_faces:
            dir_vec = Vec3(*direction) * BLOCK_SCALE
            neighbor = find_block_at(current.position + dir_vec)

            if neighbor and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited


def update_disjoint_lines():
    """Calculates and redraws boundaries between adjacent blocks that are not glued."""
    global disjoint_lines

    for line in disjoint_lines:
        destroy(line)
    disjoint_lines.clear()

    pos_map = {tuple(round(val, 4) for val in b.position): b for b in blocks_in_world}
    offsets = [
        Vec3(BLOCK_SCALE, 0, 0),
        Vec3(-BLOCK_SCALE, 0, 0),
        Vec3(0, BLOCK_SCALE, 0),
        Vec3(0, -BLOCK_SCALE, 0),
        Vec3(0, 0, BLOCK_SCALE),
        Vec3(0, 0, -BLOCK_SCALE),
    ]

    processed_pairs = set()

    for pos, block in pos_map.items():
        for d in offsets:
            neighbor_pos = tuple(round(val, 4) for val in (Vec3(*pos) + d))
            if neighbor_pos in pos_map:
                neighbor = pos_map[neighbor_pos]

                pair_id = tuple(sorted([id(block), id(neighbor)]))
                if pair_id in processed_pairs:
                    continue
                processed_pairs.add(pair_id)

                # Are they glued along this specific direction?
                dir_tuple = (
                    int(d.x / BLOCK_SCALE),
                    int(d.y / BLOCK_SCALE),
                    int(d.z / BLOCK_SCALE),
                )

                if dir_tuple not in block.glued_faces:
                    draw_boundary_line(block.position, neighbor.position, d)


def draw_boundary_line(pos1, pos2, direction):
    """Draws wireframe lines separating disjoint blocks."""
    visual_pos1 = pos1 - Vec3(0, BLOCK_SCALE / 2, 0)
    visual_pos2 = pos2 - Vec3(0, BLOCK_SCALE / 2, 0)
    midpoint = (visual_pos1 + visual_pos2) / 2
    offset = (BLOCK_SCALE / 2) * 1.01

    vertices = []
    if direction.x != 0:
        vertices = [
            midpoint + Vec3(0, -offset, -offset),
            midpoint + Vec3(0, offset, -offset),
            midpoint + Vec3(0, offset, offset),
            midpoint + Vec3(0, -offset, offset),
            midpoint + Vec3(0, -offset, -offset),
        ]
    elif direction.y != 0:
        vertices = [
            midpoint + Vec3(-offset, 0, -offset),
            midpoint + Vec3(offset, 0, -offset),
            midpoint + Vec3(offset, 0, offset),
            midpoint + Vec3(-offset, 0, offset),
            midpoint + Vec3(-offset, 0, -offset),
        ]
    elif direction.z != 0:
        vertices = [
            midpoint + Vec3(-offset, -offset, 0),
            midpoint + Vec3(offset, -offset, 0),
            midpoint + Vec3(offset, offset, 0),
            midpoint + Vec3(-offset, offset, 0),
            midpoint + Vec3(-offset, -offset, 0),
        ]

    line = Entity(
        model=Mesh(vertices=vertices, mode="line", thickness=2),
        color=color.black,
        lit=False,
    )
    disjoint_lines.append(line)


def notify_text(msg):
    """Utility to flash user notifications on screen."""
    notification.text = msg
    notification.appear_time = time.time()


# --- Scene Setup ---

# 40x40 Unglued Floor (at y=0)
# Even though they are unglued (glued_faces is empty), the Block class hardcodes
# self.y <= 0.001 to prevent falling further down.
half_floor_size = 5
for z in range(-half_floor_size, half_floor_size):
    for x in range(-half_floor_size, half_floor_size):
        Block(position=(x * BLOCK_SCALE, 0, z * BLOCK_SCALE), block_type="gravel")

# Player spawn (raised significantly to y=50 as requested)
player = FirstPersonController()
player.y = 0.0
player.gravity = 0
player.cursor.enabled = False

# Crosshair HUD
crosshair = Entity(
    parent=camera.ui,
    model="quad",
    scale=(0.015, 0.015),
    texture="crosshair",
    color=color.black,
)

# Sky and selection indicators
sky = Sky(color=color.cyan)

selection_text = Text(
    text=f"Selected: {block_types[active_block_type].upper()}",
    position=(-0.1, 0.4),
    scale=2,
    color=color.black,
)

notification = Text(text="", position=(-0.3, 0.3), scale=1.5, color=color.red)


# Notification auto-fade loop
def update():
    if (
        hasattr(notification, "appear_time")
        and time.time() - notification.appear_time > 2.0
    ):
        notification.text = ""


# Basic dynamic lighting and shadows
sun = DirectionalLight(y=20, rotation=(45, -45, 0))
sun.shadow_map_resolution = (2048, 2048)

# --- Input Handling ---

pause_manager = Entity(ignore_paused=True)


def pause_input(key):
    global active_block_type, is_paused

    if key == "escape":
        is_paused = not is_paused
        application.paused = is_paused

        if is_paused:
            mouse.locked = False
            mouse.visible = True
            player.enabled = False
            crosshair.enabled = False
            selection_text.text = "PAUSED (ESC to Resume)"
        else:
            mouse.locked = True
            mouse.visible = False
            player.enabled = True
            crosshair.enabled = True
            selection_text.text = f"Selected: {block_types[active_block_type].upper()}"

    if not is_paused:
        if key == "scroll up":
            active_block_type = (active_block_type + 1) % len(block_types)
            selection_text.text = f"Selected: {block_types[active_block_type].upper()}"

        elif key == "scroll down":
            active_block_type = (active_block_type - 1) % len(block_types)
            selection_text.text = f"Selected: {block_types[active_block_type].upper()}"


pause_manager.input = pause_input

app.run()

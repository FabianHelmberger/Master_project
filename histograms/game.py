import numpy as np
import pygame
from scipy.integrate import solve_ivp

# Define the action and its derivatives
def S(z, sigma=4j, interaction=2*6):
    """Action S(z) = 1/2 * sigma * z^2 + interaction / 4 * z^4"""
    return 1/2 * sigma * z**2 + interaction / 4 * z**4

def S_prime(z, sigma=4j, interaction=2*6):
    """Derivative of the action with respect to z."""
    return sigma * z + interaction * z**3

def crit_points(sigma, interaction):
    """Find the critical points of the action."""
    cp0 = -np.sqrt(-sigma / interaction)
    cp1 = 0
    cp2 = +np.sqrt(-sigma / interaction)
    return np.array([cp0, cp1, cp2])

def flow_equation(t, z, sigma, interaction):
    """The flow equation dz/dt = conjugate(S'(z))"""
    dz_dt = np.conj(S_prime(z, sigma, interaction))
    return dz_dt

# Function to calculate flow lines
def compute_flow_lines(sigma, interaction, t_span, t_eval, epsilon, critical_points):
    """Compute flow lines for the critical points."""
    flow_lines = []
    for cp in critical_points:
        for perturb in [+epsilon, -epsilon]:
            z0 = cp + perturb
            solution = solve_ivp(
                flow_equation, t_span, [z0], args=(sigma, interaction), t_eval=t_eval, method="RK45"
            )
            flow_lines.append(solution.y[0])
    return flow_lines

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((800, 800))
pygame.display.set_caption("Lefschetz Thimbles Flow with Mouse Control")
clock = pygame.time.Clock()

# Game parameters
interaction = 12
step = 0.2  # Step size for modifying sigma
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Reduce resolution for faster calculation
epsilon = 0.01j
grid_resolution = 100  # Increased grid resolution for better detail

# Precompute a reusable grid for vector fields
x_range, y_range = (-4, 4), (-4, 4)  # Increased range for zooming in
x = np.linspace(*x_range, grid_resolution)
y = np.linspace(*y_range, grid_resolution)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Create two slider regions for sigma and interaction
sigma_slider_rect = pygame.Rect(50, 700, 200, 40)  # Slider for sigma
interaction_slider_rect = pygame.Rect(300, 700, 200, 40)  # Slider for interaction

# Flags to track if sliders are being dragged
dragging_sigma = False
dragging_interaction = False

# Main game loop
running = True
sigma = 4j  # Initial value of sigma
while running:
    # Handle user input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the mouse click is inside the slider areas
            if sigma_slider_rect.collidepoint(event.pos):
                dragging_sigma = True
            if interaction_slider_rect.collidepoint(event.pos):
                dragging_interaction = True

        if event.type == pygame.MOUSEBUTTONUP:
            dragging_sigma = False
            dragging_interaction = False

        if event.type == pygame.MOUSEMOTION:
            # Update the sigma value when dragging the sigma slider
            if dragging_sigma:
                mouse_x, mouse_y = event.pos
                sigma_real = (mouse_x - sigma_slider_rect.left) / 200.0 * 4 - 2  # Real part range [-2, 2]
                sigma_imag = -(mouse_y - sigma_slider_rect.top) / 40.0 * 4 - 2  # Imaginary part range [-2, 2]
                sigma = sigma_real + sigma_imag * 1j

            # Update the interaction value when dragging the interaction slider
            if dragging_interaction:
                mouse_x, mouse_y = event.pos
                interaction_real = (mouse_x - interaction_slider_rect.left) / 200.0 * 4 - 2  # Real part range [-2, 2]
                interaction_imag = -(mouse_y - interaction_slider_rect.top) / 40.0 * 4 - 2  # Imaginary part range [-2, 2]
                interaction = interaction_real + interaction_imag * 1j

    # Clear screen
    screen.fill((255, 255, 255))

    # Compute critical points and flow lines
    critical_points = crit_points(sigma, interaction)
    flow_lines = compute_flow_lines(sigma, interaction, t_span, t_eval, epsilon, critical_points)

    # Draw critical points with different colors
    for cp, col in zip(critical_points, [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        pygame.draw.circle(screen, col, 
                           (int(400 + cp.real * 200), int(400 - cp.imag * 200)), 5)

    # Draw flow lines
    for line in flow_lines:
        points = [(400 + z.real * 200, 400 - z.imag * 200) for z in line]
        pygame.draw.lines(screen, (0, 0, 0), False, points, 1)

    # Compute vector field and draw arrows
    S_values = S(Z, sigma)
    U = np.real(S_values)
    V = np.imag(S_values)
    mags = np.sqrt(U**2 + V**2)*1.5

    # Draw vector field with adjusted scaling for zoom
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            x_start = 400 + X[i, j] * 200
            y_start = 400 - Y[i, j] * 200
            # Increase the length of the arrows for zooming
            scale_factor = 20  # Adjust the scaling factor for better visibility
            x_end = x_start + (U[i, j] / mags[i, j]) * scale_factor
            y_end = y_start - (V[i, j] / mags[i, j]) * scale_factor
            pygame.draw.line(screen, (0, 0, 255), (x_start, y_start), (x_end, y_end), 1)

    # Display the sigma as a text label at the bottom
    font = pygame.font.SysFont("Arial", 18)
    sigma_text = font.render(f"sigma = {sigma.real:.2f} + {sigma.imag:.2f}j", True, (0, 0, 0))
    screen.blit(sigma_text, (10, 760))

    # Draw slider rectangles
    pygame.draw.rect(screen, (200, 200, 200), sigma_slider_rect)
    pygame.draw.rect(screen, (200, 200, 200), interaction_slider_rect)

    # Update the display
    pygame.display.flip()
    clock.tick(30)

pygame.quit()

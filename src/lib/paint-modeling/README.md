# Paint Modeling

Paint modeling stores brushstrokes as continuous ribbon surfaces. A stroke owns a sparse ribbon source. Charts, surface references, chart depth, chart overlays, occlusion claims, CPU ribbon raycast, and CPU evaluated ribbon meshes are not part of the current model.

## Core State

A saved view is a calibrated camera. It records the camera matrices and viewport size used when the stroke was drawn.

A paint object owns strokes through `PaintStroke.objectId`. It no longer owns charts.

A paint stroke stores:

```ts
type PaintStroke = {
    id: string,
    objectId: string,
    sourceViewId: string,
    layerId: string,
    sourcePoints: Vec2[],
    ribbon: {
        closed: boolean,
        vertices: {
            position: Vec3,
            side: Vec3,
            u: number,
        }[],
    },
};
```

`u` runs along the stroke. Each ribbon vertex is oriented: `position` is the center point and `side` is the half-width vector across the ribbon.

## Placement

`BrushPlacementMode` exposes five intent-level presets:

- `View` uses the active view plane.
- `StartDepth` uses the first surface hit's camera-forward depth with a view-facing normal.
- `StartPlane` reuses the first surface hit's full plane.
- `Surface` uses direct surface planes and bridges unprojectable runs.
- `ConstructionPlane` uses a persistent world-space origin and normal.

Surface placement has two GPU compute stages. The first resolves direct ribbon hits in parallel. The second finds each missing sample's neighboring hits and interpolates a plane using stroke-pixel arc length and cubic Hermite endpoint derivatives. Leading and trailing misses reuse the nearest valid plane. If the stroke has no surface hit, it uses the view plane.

The construction-plane editor exposes camera-forward view depth and a normalized world-space normal. Editing depth moves the plane origin along the current view direction. The stored plane remains world-space when the camera moves.

## Stroke Commit

On pointer up, the draft path is sampled into a 2D source path. GPU placement emits one oriented ribbon vertex per centerline sample for every placement mode. The CPU reads only the completed sparse ribbon. A view-plane CPU path remains as an error fallback.

Closed source paths drop the duplicated closing source point and mark the ribbon as closed. The GPU wraps the last oriented vertex back to the first one when it expands the ribbon.

## Rendering

Committed and draft ribbons render as `RenderRibbon` primitives. They upload only oriented vertices plus color and shading uniforms.

The ribbon shader expands adjacent oriented vertices into the two ribbon edges and triangles with `vertex_index`. CPU code does not build ribbon faces, evaluated surfaces, render triangles, brush guide geometry, or raycast triangles. Draft ribbons render opaque and use a small GPU clip-depth bias so preview strokes do not fight committed surfaces.

The GPU brush guide draws the local brush ring and normal. Construction-plane mode also draws a finite major/minor grid, colored origin axes, a normal, and a faint xray pass for occluded lines.

There is no chart upload store. There are no chart wire or surface-field primitives.

## Picking

There is no committed CPU ribbon picking path. Surface placement expands committed ribbon segments and raycasts them inside WGSL. Construction-plane picking reads one GPU hover result and stores its origin and normal. Future picking should follow that GPU boundary or use a deliberate source-level interaction model, not a CPU evaluated surface.

## Current Boundaries

View-claim locking is deferred. Strokes remember their source view, but up to three claimed views are not implemented yet.

Billboard paint is not a committed surface mode. The renderer keeps generic stroke primitives for guide and legacy render paths, but committed paint strokes are ribbons.
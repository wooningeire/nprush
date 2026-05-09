/** Values for `ViewerState.renderMode` and the render-mode `<select>`. */
export const RENDER_MODE_SINGLE_VIEW_REALTIME = 'single-view-realtime' as const;
export const RENDER_MODE_MULTIVIEW = 'multiview' as const;

export type RenderMode =
    | typeof RENDER_MODE_SINGLE_VIEW_REALTIME
    | typeof RENDER_MODE_MULTIVIEW;

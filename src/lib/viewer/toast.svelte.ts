export type ToastKind = "info" | "error" | "success";

export interface Toast {
    id: number;
    message: string;
    kind: ToastKind;
}

let nextId = 0;
export const toasts = $state<Toast[]>([]);

export function showToast(message: string, kind: ToastKind = "info", duration = 4000): number {
    const id = nextId++;
    toasts.push({ id, message, kind });
    if (duration > 0) setTimeout(() => dismissToast(id), duration);
    return id;
}

export function dismissToast(id: number) {
    const idx = toasts.findIndex(t => t.id === id);
    if (idx !== -1) toasts.splice(idx, 1);
}

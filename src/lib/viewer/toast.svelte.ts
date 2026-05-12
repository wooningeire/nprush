export type ToastKind = "info" | "error" | "success";

export interface Toast {
    id: number;
    message: string;
    kind: ToastKind;
}

let nextId = 0;
export const toasts = $state<Toast[]>([]);

export function showToast(message: string, kind: ToastKind = "info", duration = 4000) {
    const id = nextId++;
    toasts.push({ id, message, kind });
    setTimeout(() => dismissToast(id), duration);
}

export function dismissToast(id: number) {
    const idx = toasts.findIndex(t => t.id === id);
    if (idx !== -1) toasts.splice(idx, 1);
}

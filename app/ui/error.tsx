import cx from "@/app/lib/cx";

export default function Error() {
    return (
        <article
            className={cx(
                "mb-2 flex items-start gap-4 rounded-2xl p-4 md:p-5",
                "bg-red-400 font-bold text-white",
            )}
        >
            No fue posible completar la acción. Vuelve a intentarlo.
        </article>
    );
}
